import logging
from typing import Dict, Any, Optional, List
from llama_index.core.indices.base import BaseIndex
from llama_index.core.schema import Document
from llama_index.core import VectorStoreIndex

logger = logging.getLogger(__name__)


class IndexMetadataExtractor:
    """
    VectorStoreIndexは未対応らしい
    "Vector store integrations that store text in the vector store are not supported by ref_doc_info yet."
    """
    @staticmethod 
    def extract_metadata(index: BaseIndex) -> Dict[str, Any]:
        metadata = {}    
        
        if isinstance(index, VectorStoreIndex):
            return IndexMetadataExtractor._extract_from_vector_index(index)
        
        metadata_from_docstore = IndexMetadataExtractor._extract_from_docstore(index)
        if metadata_from_docstore:
            metadata.update(metadata_from_docstore)
            logger.info(f"Extracted metadata from docstore: {metadata}")
            return metadata

        metadata_from_ref = IndexMetadataExtractor._extract_from_ref_doc_info(index)
        if metadata_from_ref:
            metadata.update(metadata_from_ref)
            logger.info(f"Extracted metadata from ref_doc_info: {metadata}")
            return metadata

            
        logger.warning("No metadata found in index")
        return metadata
    
    
    def _extract_from_vector_index(index: VectorStoreIndex) -> Dict[str, Any]:
        index_stuct = index.index_struct
        return {}
        
    
    @staticmethod
    def _extract_from_ref_doc_info(index: BaseIndex) -> Dict[str, Any]:
        if not hasattr(index, 'ref_doc_info') or not index.ref_doc_info:
            return {}
        
        first_doc_id = next(iter(index.ref_doc_info.keys()))
        ref_info = index.ref_doc_info[first_doc_id]
        
        if hasattr(ref_info, 'metadata'):
            return ref_info.metadata
        
        return {}
    
    
    @staticmethod
    def _extract_from_docstore(index: BaseIndex) -> Dict[str, Any]:
        if not hasattr(index, '_docstore'):
            return {}
        
        docstore = index._docstore
        doc_hashes = docstore.get_all_document_hashes()
        
        if not doc_hashes:
            return {}
        
        first_doc_id = next(iter(doc_hashes.values()))
        doc = docstore.get_document(first_doc_id)
        
        return doc.metadata if doc else {}
    
    
    @staticmethod
    def extract_all_documents_metadata(index: BaseIndex) -> List[Dict[str, Any]]:
        all_metadata = []
        
        if hasattr(index, 'ref_doc_info') and index.ref_doc_info:
            for doc_id, ref_info in index.ref_doc_info.items():
                if hasattr(ref_info, 'metadata'):
                    all_metadata.append({
                        'doc_id': doc_id,
                        'metadata': ref_info.metadata
                    })
        elif hasattr(index, '_docstore'):
            doc_hashes = index._docstore.get_all_document_hashes()
            for doc_id in doc_hashes.keys():
                doc = index._docstore.get_document(doc_id)
                if doc:
                    all_metadata.append({
                        'doc_id': doc_id,
                        'metadata': doc.metadata
                    })
        
        logger.info(f"Extract metadata from {len(all_metadata)} documents")
        return all_metadata
    
    
    @staticmethod
    def get_document_name(index: BaseIndex, default:str='unknown') -> str:
        metadata = IndexMetadataExtractor.extract_metadata(index)
        if 'name' in metadata:
            return metadata['name']
        elif 'term_name' in metadata:
            return metadata['term_name']
        elif 'file_name' in metadata:
            return metadata['file_name']
        elif 'file_path' in metadata:
            return metadata['file_path']
        return default
    