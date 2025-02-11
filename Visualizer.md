                    .nr/.neural File
                      │
                      ▼
              Parser (parser.py)         Visualization Data
    (transforms text → model_data)    (JSON for D3 visualization)
                      │                      │
                      └───────────┬──────────┘
                                  ▼
                       Visualization Server
                  (Flask/Python or pure JavaScript)
                                  │
                                  ▼
                          D3.js Frontend