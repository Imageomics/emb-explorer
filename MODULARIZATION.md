# Modularized Structure Guide

## 📁 Directory Structure

```
emb-explorer/
├── app.py                           # Main entry point (welcome page)
├── pages/                           # Streamlit pages
│   └── 01_Clustering.py            # Clustering functionality page
├── components/                      # UI components
│   └── clustering/                 # Clustering page UI components
│       ├── sidebar.py             # Sidebar controls
│       ├── visualization.py       # Scatter plot and image preview
│       └── summary.py             # Clustering summary panel
├── services/                       # Business logic services
│   ├── embedding_service.py      # Embedding generation
│   ├── clustering_service.py     # Clustering operations
│   └── file_service.py           # File save/repartition operations
├── lib/                           # Infrastructure utilities
│   └── progress.py               # Progress bar context managers
└── utils/                         # Low-level utilities (existing)
    ├── models.py
    ├── clustering.py
    └── io.py
```

## 🎯 Key Benefits

1. **Clean Separation**: UI components handle presentation, service layer handles business logic
2. **Progress Bars Preserved**: Context managers ensure progress bars work exactly as before
3. **Modular Design**: Each component has a single responsibility
4. **Future Ready**: Easy to add new pages and features
5. **Testable**: Service logic can be tested independently
6. **Clear Structure**: `utils/` for algorithms, `services/` for business logic, `components/` for UI, `lib/` for infrastructure

## 🚀 How to Run

The app now uses Streamlit's multi-page feature:

```bash
streamlit run app.py
```

- Main page: Welcome/landing page
- Clustering page: All the original functionality, now modularized

## 🔧 Key Components

### Progress Management (`lib/progress.py`)
- `StreamlitProgressContext`: Automatic progress bar cleanup with success/error handling
- `MockProgressContext`: For testing without UI

### Service Layer
- `EmbeddingService`: Handles model loading and embedding generation
- `ClusteringService`: Manages clustering workflows and summary generation  
- `FileService`: Handles save and repartition operations

### UI Components
- Page-specific components in `components/clustering/`
- Infrastructure utilities in `lib/`
- Clean separation between UI and business logic

## 🧪 Testing the Migration

1. **Functionality**: All original features should work identically
2. **Progress Bars**: Should show the same progress updates
3. **Session State**: Clustering data persists between page interactions
4. **Error Handling**: Better error messages and cleanup

## 🔄 Adding New Pages

To add a new page:

1. Create `pages/02_NewPage.py`
2. Create UI components in `components/newpage/`
3. Add services if needed in `services/`
4. Add infrastructure utilities in `lib/` if needed
5. Follow the same components/services pattern

The modular structure makes it easy to reuse components and maintain consistency across pages.
