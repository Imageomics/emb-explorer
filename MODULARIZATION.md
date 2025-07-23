# Modularized Structure Guide

## 📁 Directory Structure

```
emb-explorer/
├── app.py                           # Main entry point (welcome page)
├── pages/                           # Streamlit pages
│   └── 01_Clustering.py            # Clustering functionality page
├── ui/                             # UI components
│   ├── clustering/                 # Clustering page UI components
│   │   ├── sidebar.py             # Sidebar controls
│   │   ├── visualization.py       # Scatter plot and image preview
│   │   └── summary.py             # Clustering summary panel
│   └── shared/                    # Shared UI utilities
│       └── progress.py            # Progress bar context managers
├── server/                        # Business logic services
│   ├── embedding_service.py      # Embedding generation
│   ├── clustering_service.py     # Clustering operations
│   └── file_service.py           # File save/repartition operations
└── utils/                         # Low-level utilities (existing)
    ├── models.py
    ├── clustering.py
    └── io.py
```

## 🎯 Key Benefits

1. **Clean Separation**: UI components handle presentation, server services handle business logic
2. **Progress Bars Preserved**: Context managers ensure progress bars work exactly as before
3. **Modular Design**: Each component has a single responsibility
4. **Future Ready**: Easy to add new pages and features
5. **Testable**: Server logic can be tested independently

## 🚀 How to Run

The app now uses Streamlit's multi-page feature:

```bash
streamlit run app.py
```

- Main page: Welcome/landing page
- Clustering page: All the original functionality, now modularized

## 🔧 Key Components

### Progress Management (`ui/shared/progress.py`)
- `StreamlitProgressContext`: Automatic progress bar cleanup with success/error handling
- `MockProgressContext`: For testing without UI

### Server Services
- `EmbeddingService`: Handles model loading and embedding generation
- `ClusteringService`: Manages clustering workflows and summary generation  
- `FileService`: Handles save and repartition operations

### UI Components
- Page-specific components in `ui/clustering/`
- Shared utilities in `ui/shared/`
- Clean separation between UI and business logic

## 🧪 Testing the Migration

1. **Functionality**: All original features should work identically
2. **Progress Bars**: Should show the same progress updates
3. **Session State**: Clustering data persists between page interactions
4. **Error Handling**: Better error messages and cleanup

## 🔄 Adding New Pages

To add a new page:

1. Create `pages/02_NewPage.py`
2. Create UI components in `ui/newpage/`
3. Add server services if needed
4. Follow the same UI/Server pattern

The modular structure makes it easy to reuse components and maintain consistency across pages.
