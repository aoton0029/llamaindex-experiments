import logging
import json
from typing import List, Dict, Any
from pymilvus import FieldSchema, CollectionSchema, DataType

logger = logging.getLogger(__name__)

class SchemaBuilder:
    @staticmethod
    def build_schema(schema_config: List[Dict[str, Any]], dim: int) -> CollectionSchema:
        fields = [
            FieldSchema(name="_id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=255),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
        ]
        for meta_key, meta_cfg in schema_config.get("schema", {}).items():
            dtype = SchemaBuilder.convert_dtype(meta_cfg.get("dtype", "VARCHAR"))
            description = meta_cfg.get("description", "")
            if dtype == DataType.VARCHAR:
                max_length = meta_cfg.get("max_length", 255)
                fields.append(
                    FieldSchema(name=meta_key, description=description, dtype=dtype, max_length=max_length)
                )
            else:
                fields.append(
                    FieldSchema(name=meta_key, description=description, dtype=dtype)
                )
        schema = CollectionSchema(fields, description="Embedding collection schema")
        return schema
    
    @staticmethod
    def build_metadata_field(metadata: Dict[str, Any], metadata_config: List[Dict[str, Any]]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for md_cfg in metadata_config:
            key = md_cfg.get("name")
            out[key] = metadata.get(key, "")
        return out

    @staticmethod
    def convert_dtype(dtype_str: str) -> DataType:
        dtype_map = {
            "INT64": DataType.INT64,
            "FLOAT_VECTOR": DataType.FLOAT_VECTOR,
            "VARCHAR": DataType.VARCHAR,
        }
        return dtype_map.get(dtype_str.upper(), DataType.VARCHAR)

