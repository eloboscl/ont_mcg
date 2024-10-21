import pandas as pd
import json
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class MetadataIntegrator:
    def __init__(self, metadata_file: str):
        self.metadata = pd.read_excel(metadata_file)
        logger.info(f"Loaded metadata from {metadata_file}")

    def integrate_metadata(self, documents: Dict[str, Any]) -> Dict[str, Any]:
        integrated_documents = {}
        for key, doc in documents.items():
            metadata_row = self.metadata[self.metadata['document_name'] == key]
            if not metadata_row.empty:
                integrated_doc = doc.copy()
                for column in metadata_row.columns:
                    if column != 'document_name':
                        integrated_doc[column] = metadata_row[column].values[0]
                integrated_documents[key] = integrated_doc
                logger.info(f"Integrated metadata for document: {key}")
            else:
                logger.warning(f"No metadata found for document: {key}")
                integrated_documents[key] = doc
        return integrated_documents

def main(documents_file: str, metadata_file: str, output_file: str):
    logger.info("Starting metadata integration process")

    # Load documents
    with open(documents_file, 'r', encoding='utf-8') as f:
        documents = json.load(f)

    # Initialize integrator
    integrator = MetadataIntegrator(metadata_file)

    # Integrate metadata
    integrated_documents = integrator.integrate_metadata(documents)

    # Save integrated documents
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(integrated_documents, f, ensure_ascii=False, indent=4)

    logger.info(f"Integration complete. Integrated documents saved to: {output_file}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 4:
        print("Usage: python metadata_integrator.py <documents_file> <metadata_file> <output_file>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2], sys.argv[3])