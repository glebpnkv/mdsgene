import React, { useState, useCallback, useEffect, useRef } from 'react';
// Using lucide-react for icons
import { Upload, Loader2, FileText, CheckCircle, XCircle, Send, FileSearch, RefreshCw, Trash2, AlertTriangle, Ban, ChevronDown, Users, User, ServerCrash } from 'lucide-react'; // Added ServerCrash for Reset

// Configuration for the backend API URL
const API_BASE_URL = 'http://34.147.75.119:8000';

// --- Helper Components ---
const StatusBar = ({ message, type = 'status' }) => {
  if (!message) return null;
  const baseClasses = "p-3 rounded-md text-sm font-medium mb-4";
  const typeClasses = {
    status: "bg-blue-100 text-blue-800",
    success: "bg-green-100 text-green-800",
    error: "bg-red-100 text-red-800",
    warning: "bg-yellow-100 text-yellow-800",
  };
  return (
    <div className={`${baseClasses} ${typeClasses[type] || typeClasses.status}`}>
      {message}
    </div>
  );
};

// --- Main Application Component ---
function App() {
  // State Variables
  const [selectedFile, setSelectedFile] = useState(null);
  const [description, setDescription] = useState('');
  const [documents, setDocuments] = useState([]); // Basic doc info { id, filename, description }
  const [singleResult, setSingleResult] = useState(null); // {field, patient_id, family_id, value, raw_answer?}
  const [confirmingDeleteId, setConfirmingDeleteId] = useState(null); // For single doc delete confirmation
  const [confirmingResetAll, setConfirmingResetAll] = useState(false); // Added state for reset confirmation
  const [availableMappingFields, setAvailableMappingFields] = useState([]); // For field dropdown
  const [selectedMappingField, setSelectedMappingField] = useState(''); // Selected field
  const [selectedDocIdForPatientLoad, setSelectedDocIdForPatientLoad] = useState(null); // Track which doc we loaded patients for
  const [patientsForSelectedDoc, setPatientsForSelectedDoc] = useState(null); // Patient list for the selected doc {family_id, patient_id, display_name}[]
  const [selectedPatientKey, setSelectedPatientKey] = useState(''); // Key like "F1_P1" or "NOFAMILY_P2" for patient dropdown
  const [isLoadingPatients, setIsLoadingPatients] = useState(false); // Loading state for patients

  // Loading and Status States
  const [isUploading, setIsUploading] = useState(false);
  const [isLoadingInitialData, setIsLoadingInitialData] = useState(true);
  const [isBackendProcessing, setIsBackendProcessing] = useState(false); // Global processing flag (for single field or patient fetch)
  const [isRequestingCancel, setIsRequestingCancel] = useState(false);
  const [isDeleting, setIsDeleting] = useState({});
  const [isResetting, setIsResetting] = useState(false); // Added state for reset operation
  const [statusMessage, setStatusMessage] = useState('');
  const [errorMessage, setErrorMessage] = useState('');

  // --- Fetch Initial Data (Docs List, Status, and Mapping Items) ---
  const fetchInitialData = useCallback(async (showLoading = true) => {
    if (showLoading) setIsLoadingInitialData(true);
    setErrorMessage('');
    let loadedDocs = [];
    let processingStatus = false;
    let mappingFields = [];
    let statusMsg = '';
    let errorMsg = '';
    try {
        // Fetch documents, status, and mapping items in parallel
        const [docsResponse, statusResponse, itemsResponse] = await Promise.all([
            fetch(`${API_BASE_URL}/documents`),
            fetch(`${API_BASE_URL}/processing_status`),
            fetch(`${API_BASE_URL}/mapping_items`)
        ]);
        // Process documents response
        if (!docsResponse.ok) { throw new Error(`Docs fetch failed: ${docsResponse.status}`); }
        const docsFromServer = await docsResponse.json();
        loadedDocs = docsFromServer.map(doc => ({ ...doc, status: 'uploaded' }));
        // Process status response
        if (!statusResponse.ok) { throw new Error(`Status fetch failed: ${statusResponse.status}`); }
        const statusResult = await statusResponse.json();
        processingStatus = statusResult.is_processing;
        // Process mapping items response
        if (!itemsResponse.ok) { throw new Error(`Mapping items fetch failed: ${itemsResponse.status}`); }
        mappingFields = await itemsResponse.json();
        // Set status message
        statusMsg = loadedDocs.length > 0 ? `Loaded ${loadedDocs.length} document(s).` : 'No existing documents found.';
        if (processingStatus) { statusMsg += ' A mapping process is currently running.'; }
    } catch (error) {
        console.error("Failed to fetch initial data:", error);
        errorMsg = `Failed to load initial data: ${error.message}`;
        loadedDocs = []; mappingFields = []; processingStatus = false;
    } finally {
        setDocuments(loadedDocs);
        setIsBackendProcessing(processingStatus);
        setAvailableMappingFields(mappingFields);
        if (mappingFields.length > 0 && !selectedMappingField) {
             setSelectedMappingField(mappingFields[0].field);
        } else if (mappingFields.length === 0) { setSelectedMappingField(''); }
        if (selectedDocIdForPatientLoad !== null && !loadedDocs.some(d => d.id === selectedDocIdForPatientLoad)) {
            setSelectedDocIdForPatientLoad(null); setPatientsForSelectedDoc(null); setSelectedPatientKey('');
        }
        // Only set status message if we were showing the main loading indicator
        if (showLoading) setStatusMessage(statusMsg);
        setErrorMessage(errorMsg);
        if (showLoading) setIsLoadingInitialData(false);
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedDocIdForPatientLoad]); // Removed selectedMappingField dependency

  useEffect(() => {
    fetchInitialData();
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []); // Run only once on mount

  // --- Event Handlers ---
  const handleFileChange = (event) => { setSelectedFile(event.target.files[0] || null); clearMessages(); };
  const handleDescriptionChange = (event) => { setDescription(event.target.value); };
  const handleMappingFieldChange = (event) => { setSelectedMappingField(event.target.value); setSingleResult(null); clearMessages(); };
  const handlePatientSelectionChange = (event) => { setSelectedPatientKey(event.target.value); setSingleResult(null); clearMessages(); };
  const clearMessages = () => { setStatusMessage(''); setErrorMessage(''); };
  const requestDeleteConfirmation = (docId) => { if (isBackendProcessing) return; setConfirmingDeleteId(docId); clearMessages(); };
  const cancelDeleteConfirmation = () => { setConfirmingDeleteId(null); };
  const requestResetAllConfirmation = () => { if (isBackendProcessing) return; setConfirmingResetAll(true); clearMessages(); }; // Added
  const cancelResetAllConfirmation = () => { setConfirmingResetAll(false); }; // Added

  // --- API Call Functions ---
  const handleUpload = useCallback(async () => {
     if (!selectedFile) { setErrorMessage('Please select a file.'); return; }
     clearMessages(); setIsUploading(true); setSingleResult(null);
     const formData = new FormData(); formData.append('file', selectedFile); if (description) { formData.append('description', description); }
     try {
         const response = await fetch(`${API_BASE_URL}/upload_document`, { method: 'POST', body: formData });
         const result = await response.json(); if (!response.ok) { throw new Error(result.detail || `Upload failed: ${response.status}`); }
         await fetchInitialData(false); setStatusMessage(`'${result.filename}' uploaded (ID: ${result.assigned_doc_id}).`);
         setSelectedFile(null); setDescription(''); const fileInput = document.getElementById('file-upload'); if (fileInput) fileInput.value = '';
     } catch (error) { console.error('Upload error:', error); setErrorMessage(`Upload failed: ${error.message}`); await fetchInitialData(false); }
     finally { setIsUploading(false); }
  }, [selectedFile, description, fetchInitialData]);

  const handleLoadPatients = useCallback(async (docId) => {
      clearMessages(); setIsLoadingPatients(true); setSelectedDocIdForPatientLoad(docId);
      setPatientsForSelectedDoc(null); setSelectedPatientKey(''); setSingleResult(null);
      try {
          const response = await fetch(`${API_BASE_URL}/documents/${docId}/patients`);
          if (!response.ok) { let eD=`Load patients failed: ${response.status}`; try{const eR=await response.json();eD=eR.detail||eD;}catch(e){} throw new Error(eD); }
          const pList = await response.json(); if (!Array.isArray(pList)) { throw new Error("Invalid patient data."); }
          setPatientsForSelectedDoc(pList);
          if (pList.length > 0) { const fP=pList[0]; setSelectedPatientKey(`${fP.family_id||'NOFAMILY'}_${fP.patient_id}`); setStatusMessage(`Loaded ${pList.length} patients for doc ID ${docId}.`); }
          else { setStatusMessage(`No patients found in doc ID ${docId}.`); }
      } catch (error) { console.error("Load patients error:", error); setErrorMessage(`Load patients failed: ${error.message}`); }
      finally { setIsLoadingPatients(false); }
  }, []);

  const handleProcessPatientField = useCallback(async (docId) => {
    if (isBackendProcessing) { setErrorMessage(`Another process is running.`); return; }
    if (docId === null || docId === undefined) { setErrorMessage('Invalid document ID.'); return; }
    if (!selectedMappingField) { setErrorMessage('Select a field to process.'); return; }
    if (!selectedPatientKey) { setErrorMessage('Select a patient to process.'); return; }
    if (selectedDocIdForPatientLoad !== docId || !patientsForSelectedDoc) { setErrorMessage('Load patients for this document first.'); return;}
    const selectedPatientObj = patientsForSelectedDoc.find(p => `${p.family_id || 'NOFAMILY'}_${p.patient_id}` === selectedPatientKey);
    if (!selectedPatientObj) { setErrorMessage('Selected patient data not found.'); return; }

    clearMessages(); setIsBackendProcessing(true); setSingleResult(null);
    try {
      const response = await fetch(`${API_BASE_URL}/documents/${docId}/process_patient_field`, {
          method: 'POST', headers: { 'Content-Type': 'application/json', 'Accept': 'application/json' },
          body: JSON.stringify({ field: selectedMappingField, patient_id: selectedPatientObj.patient_id, family_id: selectedPatientObj.family_id })
      });
      const result = await response.json();
      if (!response.ok) {
         if (response.status === 400 && result.detail?.includes("cancelled")) { setStatusMessage(`Processing cancelled.`); }
         else { throw new Error(result.detail || `Processing failed (Status: ${response.status})`); }
      } else { setSingleResult(result); setStatusMessage(`Processing complete for field '${result.field}'.`); }
    } catch (error) { console.error('Patient field processing error:', error); setErrorMessage(error.message); }
    finally { await fetchInitialData(false); }
  }, [isBackendProcessing, fetchInitialData, selectedMappingField, selectedPatientKey, patientsForSelectedDoc, selectedDocIdForPatientLoad]);

  const handleRequestCancel = useCallback(async () => {
      clearMessages(); setIsRequestingCancel(true);
      try {
          const response = await fetch(`${API_BASE_URL}/cancel_current_mapping`, { method: 'POST' });
          const result = await response.json(); if (!response.ok) { throw new Error(result.detail || `Cancel failed: ${response.status}`); }
          setStatusMessage(result.message || "Cancellation requested."); await fetchInitialData(false);
      } catch (error) { console.error('Cancel request error:', error); setErrorMessage(`Cancel failed: ${error.message}`); await fetchInitialData(false); }
      finally { setIsRequestingCancel(false); }
  }, [fetchInitialData]);

  const executeDeleteDocument = useCallback(async (docId, docFilename) => {
       if (docId === null || docId === undefined) return; clearMessages(); setConfirmingDeleteId(null); setIsDeleting(prev => ({ ...prev, [docId]: true }));
       try {
           const response = await fetch(`${API_BASE_URL}/documents/${docId}`, { method: 'DELETE' });
           if (!response.ok) { let eD = `Delete failed: ${response.status}`; try { const eR = await response.json(); eD = eR.detail || eD; } catch (e) {} throw new Error(eD); }
           setStatusMessage(`Doc "${docFilename}" deleted.`);
           if (selectedDocIdForPatientLoad === docId) { setSelectedDocIdForPatientLoad(null); setPatientsForSelectedDoc(null); setSelectedPatientKey(''); }
           await fetchInitialData(false); setSingleResult(null);
       } catch (error) { console.error('Delete error:', error); setErrorMessage(`Delete failed: ${error.message}`); await fetchInitialData(false); }
       finally { setIsDeleting(prev => ({ ...prev, [docId]: false })); }
  }, [fetchInitialData, selectedDocIdForPatientLoad]);

  // --- Reset All Documents Handler ---
  const handleResetAll = useCallback(async () => {
       // REMOVED: Redundant window.confirm - rely on custom UI confirmation only
       // if (!window.confirm("Are you absolutely sure...?")) {
       //     setConfirmingResetAll(false);
       //     return;
       // }

       clearMessages();
       setIsResetting(true); // Set resetting state
       setConfirmingResetAll(false); // Close confirmation UI

       try {
           const response = await fetch(`${API_BASE_URL}/documents/reset`, {
               method: 'POST',
           });

           const result = await response.json(); // Attempt to parse JSON even for errors

           if (!response.ok) {
                throw new Error(result.detail || `Reset failed with status: ${response.status}`);
           }

           setStatusMessage(result.message || "All documents have been reset successfully.");
           // Clear all local state related to documents and processing
           setDocuments([]);
           setPatientsForSelectedDoc(null);
           setSelectedPatientKey('');
           setSelectedDocIdForPatientLoad(null);
           setSingleResult(null);
           setIsBackendProcessing(false); // Reset global processing flag locally
           // Optionally call fetchInitialData(false) again, though lists should be empty
           await fetchInitialData(false);


       } catch (error) {
           console.error('Reset error:', error);
           // Try to get the detailed error message
           if (error.response) {
               console.error('Error response:', await error.response.json());
           }
           setErrorMessage(`Failed to reset documents: ${error.message}`);
           // Refresh state even on error
           await fetchInitialData(false);
       } finally {
           setIsResetting(false); // Clear resetting state
       }
  }, [fetchInitialData]); // Depends on fetchInitialData


  // --- Rendering ---
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-100 to-blue-100 p-4 sm:p-8 font-sans">
      <div className="max-w-4xl mx-auto bg-white rounded-xl shadow-lg p-6 sm:p-8">
        <h1 className="text-3xl font-bold text-center text-slate-800 mb-8">
          Document Mapping Client
        </h1>

        {/* Status and Error Messages */}
        <StatusBar message={statusMessage} type="success" />
        <StatusBar message={errorMessage} type="error" />

        {/* Global Processing Indicator & Cancel Button */}
        {isBackendProcessing && ( /* ... remains the same ... */
             <div className="mb-4 p-4 border border-orange-300 bg-orange-50 rounded-lg flex items-center justify-between">
                <div className="flex items-center"> <Loader2 className="animate-spin mr-3 h-5 w-5 text-orange-600" /> <span className="text-sm font-medium text-orange-800">Processing...</span> </div>
                <button onClick={handleRequestCancel} disabled={isRequestingCancel} title="Request Cancellation" className="inline-flex items-center px-3 py-1.5 border border-transparent text-xs font-medium rounded shadow-sm text-white bg-orange-600 hover:bg-orange-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-orange-500 disabled:opacity-50">
                    {isRequestingCancel ? <Loader2 className="animate-spin h-4 w-4 mr-1" /> : <Ban className="h-4 w-4 mr-1" /> } Request Cancel
                </button>
            </div>
        )}

        {/* File Upload Section */}
        <div className="mb-8 p-6 border border-slate-200 rounded-lg bg-slate-50">
             <h2 className="text-xl font-semibold text-slate-700 mb-4">1. Upload Document</h2>
             {/* ... Upload inputs and button ... */}
             <div className="space-y-4">
                <div> <label htmlFor="file-upload" className="block text-sm font-medium text-slate-700 mb-1">Select File:</label> <input id="file-upload" type="file" accept=".pdf,.txt,.md" onChange={handleFileChange} className="block w-full text-sm text-slate-500 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100 disabled:opacity-50" disabled={isUploading || isBackendProcessing} /> {selectedFile && <p className="text-xs text-slate-500 mt-1">Selected: {selectedFile.name}</p>} </div>
                <div> <label htmlFor="description" className="block text-sm font-medium text-slate-700 mb-1">Description:</label> <input id="description" type="text" value={description} onChange={handleDescriptionChange} placeholder="Optional description..." className="block w-full px-3 py-2 border border-slate-300 rounded-md shadow-sm placeholder-slate-400 focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm disabled:opacity-50" disabled={isUploading || isBackendProcessing} /> </div>
                <button onClick={handleUpload} disabled={!selectedFile || isUploading || isBackendProcessing} className="w-full inline-flex items-center justify-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 disabled:opacity-50 disabled:cursor-not-allowed"> {isUploading ? <><Loader2 className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" /> Uploading...</> : <><Upload className="-ml-1 mr-2 h-5 w-5" /> Upload Document</>} </button>
             </div>
        </div>

        {/* Document List and Processing Trigger */}
        <div className="mb-8 p-6 border border-slate-200 rounded-lg">
             {/* ... Header + Refresh Button ... */}
             <div className="flex justify-between items-center mb-4">
                 <h2 className="text-xl font-semibold text-slate-700">2. Select Document, Patient & Field</h2>
                 <button onClick={() => fetchInitialData()} title="Refresh List & Status" disabled={isLoadingInitialData || isBackendProcessing || Object.values(isDeleting).some(v => v)} className="p-1.5 text-slate-500 hover:text-blue-600 hover:bg-slate-100 rounded-md disabled:opacity-50">
                     <RefreshCw className={`h-4 w-4 ${isLoadingInitialData ? 'animate-spin' : ''}`} />
                 </button>
            </div>
            {/* ... Loading Indicator ... */}
            {isLoadingInitialData ? ( <div className="flex justify-center items-center p-4"> <Loader2 className="animate-spin mr-3 h-5 w-5 text-blue-600" /> <span className="text-slate-600">Loading...</span> </div> )
            : documents.length > 0 ? (
                <ul className="space-y-3">
                    {/* ... Document List Item Mapping ... */}
                    {documents.map((doc) => {
                        const isCurrentDeleting = isDeleting[doc.id];
                        const isAnyDeleting = Object.values(isDeleting).some(v => v);
                        const isPatientListForThisDoc = selectedDocIdForPatientLoad === doc.id;

                        return (
                            <li key={doc.id} className={`p-3 bg-white border rounded-md shadow-sm transition-all duration-150 ${isPatientListForThisDoc ? 'border-blue-300 ring-2 ring-blue-200' : 'border-slate-200'}`}>
                                {/* Document Info + Load/Delete Buttons */}
                                <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between mb-3">
                                    {/* ... Doc Info ... */}
                                     <div className="flex items-center mr-4 flex-grow min-w-0"> <FileText className="h-5 w-5 text-slate-500 mr-3 flex-shrink-0" /> <div className="flex-grow min-w-0"> <p className="text-sm font-medium text-slate-800 truncate" title={doc.filename}>{doc.filename} (ID: {doc.id})</p> {doc.description && <p className="text-xs text-slate-500 italic truncate" title={doc.description}>{doc.description}</p>} </div> </div>
                                    {/* ... Load/Delete Buttons ... */}
                                     <div className="flex items-center space-x-2 w-full sm:w-auto justify-end flex-shrink-0 mt-2 sm:mt-0">
                                        <button onClick={() => handleLoadPatients(doc.id)} disabled={isBackendProcessing || isLoadingPatients || isAnyDeleting || confirmingDeleteId === doc.id} title="Load patients" className="inline-flex items-center px-3 py-1.5 border border-slate-300 text-xs font-medium rounded shadow-sm text-slate-700 bg-white hover:bg-slate-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 disabled:opacity-50 disabled:cursor-not-allowed"> {isLoadingPatients && selectedDocIdForPatientLoad === doc.id ? <Loader2 className="animate-spin h-4 w-4" /> : <Users className="h-4 w-4" />} <span className="ml-2 hidden sm:inline">Load Patients</span><span className="ml-2 sm:hidden">Patients</span> </button>
                                        <button onClick={() => requestDeleteConfirmation(doc.id)} disabled={isBackendProcessing || isCurrentDeleting || confirmingDeleteId === doc.id || isAnyDeleting || isLoadingPatients} title="Delete Document" className="inline-flex items-center p-1.5 border border-transparent text-xs font-medium rounded shadow-sm text-white bg-red-600 hover:bg-red-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-red-500 disabled:opacity-50 disabled:cursor-not-allowed"> {isCurrentDeleting ? <Loader2 className="animate-spin h-4 w-4" /> : <Trash2 className="h-4 w-4" />} </button>
                                     </div>
                                </div>
                                {/* Delete Confirmation UI */}
                                {confirmingDeleteId === doc.id && ( /* ... delete confirmation UI ... */
                                    <div className="mt-2 p-3 border border-red-200 bg-red-50 rounded flex items-center justify-end space-x-3"> <span className='text-xs text-red-700 font-semibold mr-2'>Delete?</span> <button onClick={() => executeDeleteDocument(doc.id, doc.filename)} disabled={isCurrentDeleting} title="Confirm Delete" className="inline-flex items-center px-3 py-1.5 border border-red-600 text-xs font-medium rounded shadow-sm text-red-600 bg-white hover:bg-red-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-red-500 disabled:opacity-50 disabled:cursor-not-allowed"> {isCurrentDeleting ? <Loader2 className="animate-spin h-4 w-4 text-red-600" /> : <CheckCircle className="h-4 w-4 text-red-600" />} <span className="ml-1">Confirm</span> </button> <button onClick={cancelDeleteConfirmation} disabled={isCurrentDeleting} title="Cancel Delete" className="inline-flex items-center px-3 py-1.5 border border-gray-300 text-xs font-medium rounded shadow-sm text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 disabled:opacity-50"> <XCircle className="h-4 w-4 text-gray-600" /> <span className="ml-1">Cancel</span> </button> </div>
                                )}
                                {/* Patient/Field Selector and Process Button */}
                                {isPatientListForThisDoc && patientsForSelectedDoc && !confirmingDeleteId && ( /* ... patient/field selectors and process button ... */
                                    <div className="mt-3 pt-3 border-t border-slate-200 flex flex-col sm:flex-row items-center space-y-2 sm:space-y-0 sm:space-x-3">
                                        {/* Patient Dropdown */}
                                        <div className="relative flex-grow w-full sm:w-auto"> <label htmlFor={`patient-select-${doc.id}`} className="sr-only">Select Patient</label> <select id={`patient-select-${doc.id}`} value={selectedPatientKey} onChange={handlePatientSelectionChange} disabled={isBackendProcessing || patientsForSelectedDoc.length === 0} className="block w-full appearance-none bg-white border border-slate-300 text-slate-700 py-2 px-3 pr-8 rounded-md leading-tight focus:outline-none focus:bg-white focus:border-blue-500 focus:ring-1 focus:ring-blue-500 disabled:opacity-50 text-sm"> <option value="">-- Select Patient --</option> {patientsForSelectedDoc.length === 0 && <option value="" disabled>No patients found</option>} {patientsForSelectedDoc.map((p) => (<option key={`${p.family_id || 'NOFAMILY'}_${p.patient_id}`} value={`${p.family_id || 'NOFAMILY'}_${p.patient_id}`}>{p.display_name}</option>))} </select> <div className="pointer-events-none absolute inset-y-0 right-0 flex items-center px-2 text-slate-700"><Users className="h-4 w-4" /></div> </div>
                                        {/* Mapping Field Dropdown */}
                                        <div className="relative flex-grow w-full sm:w-auto"> <label htmlFor={`mapping-field-select-${doc.id}`} className="sr-only">Select Field</label> <select id={`mapping-field-select-${doc.id}`} value={selectedMappingField} onChange={handleMappingFieldChange} disabled={isBackendProcessing || availableMappingFields.length === 0} className="block w-full appearance-none bg-white border border-slate-300 text-slate-700 py-2 px-3 pr-8 rounded-md leading-tight focus:outline-none focus:bg-white focus:border-blue-500 focus:ring-1 focus:ring-blue-500 disabled:opacity-50 text-sm"> <option value="">-- Select Field --</option> {availableMappingFields.length === 0 && <option value="" disabled>No fields available</option>} {availableMappingFields.map((item) => (<option key={item.field} value={item.field} title={item.question}>{item.field} {item.column ? `(${item.column})` : ''}</option>))} </select> <div className="pointer-events-none absolute inset-y-0 right-0 flex items-center px-2 text-slate-700"><ChevronDown className="h-4 w-4" /></div> </div>
                                        {/* Process Patient Field Button */}
                                        <button onClick={() => handleProcessPatientField(doc.id)} disabled={isBackendProcessing || !selectedPatientKey || !selectedMappingField || isAnyDeleting} title={selectedPatientKey && selectedMappingField ? `Process field '${selectedMappingField}' for selected patient` : 'Select patient and field'} className="w-full sm:w-auto inline-flex items-center justify-center px-3 py-2 border border-transparent text-xs font-medium rounded shadow-sm text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 disabled:opacity-50 disabled:cursor-not-allowed"> <FileSearch className="-ml-0.5 mr-2 h-4 w-4" /> Process </button>
                                    </div>
                                )}
                            </li>
                        )
                    })}
                </ul>
            ) : ( <p className="text-sm text-center text-slate-500 py-4">No documents uploaded.</p> )}
        </div>

        {/* Single Mapping Result Section */}
        {singleResult && ( /* ... remains the same ... */
          <div className="p-6 border border-green-300 rounded-lg bg-green-50"> <h2 className="text-xl font-semibold text-green-800 mb-4">Last Processed Field Result</h2> <div className="space-y-2 text-sm"> <p><strong className="font-medium text-slate-700">Field:</strong> {singleResult.field}</p> <p><strong className="font-medium text-slate-700">Patient:</strong> {singleResult.patient_id} {singleResult.family_id ? `(Family: ${singleResult.family_id})` : '(Sporadic)'}</p> <p><strong className="font-medium text-slate-700">Value:</strong> <span className="font-mono bg-slate-100 px-1 py-0.5 rounded text-slate-800">{singleResult.value}</span></p> {singleResult.raw_answer && ( <details className="text-xs text-slate-600"> <summary className="cursor-pointer hover:text-slate-800">Show Raw Answer</summary> <pre className="mt-1 p-2 bg-slate-100 rounded overflow-auto max-h-40 text-xs">{singleResult.raw_answer}</pre> </details> )} </div> </div>
        )}

        {/* --- NEW: Reset All Section --- */}
        <div className="mt-10 pt-6 border-t border-dashed border-red-300">
             <h2 className="text-lg font-semibold text-red-700 mb-3">Danger Zone</h2>
             {!confirmingResetAll ? (
                <button
                    onClick={requestResetAllConfirmation}
                    disabled={isBackendProcessing || isResetting || Object.values(isDeleting).some(v => v)}
                    className="inline-flex items-center px-4 py-2 border border-red-600 text-sm font-medium rounded-md shadow-sm text-red-700 bg-white hover:bg-red-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-red-500 disabled:opacity-50"
                >
                    <ServerCrash className="h-5 w-5 mr-2" />
                    Reset All Documents...
                </button>
             ) : (
                // --- Reset Confirmation UI ---
                <div className="p-4 border border-red-300 bg-red-50 rounded-lg flex flex-col sm:flex-row items-center justify-between space-y-3 sm:space-y-0">
                    <p className="text-sm font-medium text-red-800 flex items-center">
                         <AlertTriangle className="h-5 w-5 mr-2 text-red-600"/>
                         Are you sure? This will delete ALL documents permanently.
                    </p>
                    <div className="flex space-x-3 flex-shrink-0">
                        <button
                            onClick={handleResetAll}
                            disabled={isResetting}
                            className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-red-600 hover:bg-red-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-red-500 disabled:opacity-50"
                        >
                            {isResetting ? <Loader2 className="animate-spin h-5 w-5 mr-2" /> : <Trash2 className="h-5 w-5 mr-2" /> }
                            Yes, Reset All
                        </button>
                         <button
                            onClick={cancelResetAllConfirmation}
                            disabled={isResetting}
                            className="inline-flex items-center px-4 py-2 border border-gray-300 text-sm font-medium rounded-md shadow-sm text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 disabled:opacity-50"
                        >
                            Cancel
                        </button>
                    </div>
                </div>
             )}
        </div>

      </div>
    </div>
  );
}

export default App;
