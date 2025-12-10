from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from pymilvus import CollectionSchema, FieldSchema, DataType
from abc import ABC, abstractmethod


class SchemaBaseModel(ABC, BaseModel):
    @abstractmethod
    def get_milvus_schema(dim:int) -> CollectionSchema:
        raise NotImplementedError()
