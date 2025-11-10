from pymilvus import FieldSchema, CollectionSchema, DataType

class SchemaBuilder:
    @staticmethod
    def build_schema() -> CollectionSchema:
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1536),
            FieldSchema(name="metadata", dtype=DataType.VARCHAR, max_length=65535),
        ]
        schema = CollectionSchema(fields, description="Embedding collection schema")
        return schema

    def build_metadata_field(self, metadata: dict) -> dict:
        # メタデータを適切な形式に変換
        return {
            "title": metadata.get("title", ""),
            "author": metadata.get("author", ""),
            "description": metadata.get("description", ""),
        }
