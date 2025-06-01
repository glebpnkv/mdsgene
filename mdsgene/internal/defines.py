NO_INFORMATION_LIST: list[str] = [
    "",
    "-",
    "couldn't find",
    "don't know",
    "information not found",
    "n/a",
    "none",
    "not applicable",
    "not mentioned",
    "not reported",
    "not specified",
    "not stated",
    "null"
    "unknown",
]

DEFAULT_SAFETY_SETTINGS = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]

MOTOR_TERM_LIST: list[str] = [
    "bradykinesia",
    "dyskinesia",
    "dystonia"
    "gait",
    "motor",
    "postural",
    "rigid",
    "tremor",
]

NON_MOTOR_TERM_LIST: list[str] = [
    "anxiety",
    "autonomic",
    "cognitive",
    "depression",
    "olfaction"
    "psychotic",
    "sleep",
]
