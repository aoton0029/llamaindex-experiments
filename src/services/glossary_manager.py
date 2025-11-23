import pandas as pd
from typing import Optional, List
from pathlib import Path
from ..models.glossary import GlossaryTerm
import json

class GlossaryDataManager:
    def __init__(self, file_path: str):
        """
        Initialize the GlossaryDataManager.
        
        Args:
            file_path: Path to the JSONL file
        """
        self.file_path = Path(file_path)
        self.terms: List[GlossaryTerm] = []
        
    def load(self) -> List[GlossaryTerm]:
        """Load glossary terms from JSONL file."""
        if not self.file_path.exists():
            self.terms = []
            return self.terms
            
        self.terms = []
        with open(self.file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    self.terms.append(GlossaryTerm(**data))
        return self.terms
    
    def save(self) -> None:
        """Save glossary terms to JSONL file."""
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.file_path, 'w', encoding='utf-8') as f:
            for term in self.terms:
                json.dump(term.dict(), f, ensure_ascii=False)
                f.write('\n')
    
    def add_term(self, term: GlossaryTerm) -> None:
        """Add a new glossary term."""
        if any(t.term == term.term for t in self.terms):
            raise ValueError(f"Term '{term.term}' already exists")
        self.terms.append(term)
        self.save()
    
    def update_term(self, term_name: str, updated_term: GlossaryTerm) -> None:
        """Update an existing glossary term."""
        for i, term in enumerate(self.terms):
            if term.term == term_name:
                self.terms[i] = updated_term
                self.save()
                return
        raise ValueError(f"Term '{term_name}' not found")
    
    def delete_term(self, term_name: str) -> None:
        """Delete a glossary term."""
        original_length = len(self.terms)
        self.terms = [t for t in self.terms if t.term != term_name]
        if len(self.terms) == original_length:
            raise ValueError(f"Term '{term_name}' not found")
        self.save()
    
    def get_term(self, term_name: str) -> Optional[GlossaryTerm]:
        """Get a specific glossary term."""
        for term in self.terms:
            if term.term == term_name:
                return term
        return None
    
    def search_terms(self, query: str) -> List[GlossaryTerm]:
        """Search terms by query string in term or definition."""
        query_lower = query.lower()
        return [
            term for term in self.terms
            if query_lower in term.term.lower() or 
               query_lower in term.definition.lower()
        ]
    
    def get_all_terms(self) -> List[GlossaryTerm]:
        """Get all glossary terms."""
        return self.terms
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert glossary terms to pandas DataFrame."""
        if not self.terms:
            return pd.DataFrame()
        return pd.DataFrame([term.dict() for term in self.terms])
