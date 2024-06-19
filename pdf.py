import os
from llama_index.core import StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.readers.file import PDFReader


#Creat or load a vector store index
def get_index(data,index_name):
    index = None
    if not os.path.exists(index_name):
        print("building index", index_name)
        index = VectorStoreIndex.from_documents(data, show_progress=True)
        index.storage_context.persist(persist_dir=index_name)
    else:
        index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=index_name)
        )
    
    return index

#Create an index and a new query engine for the index
pdf_path = os.path.join("data", "PropertyAct.pdf")
property_act_pdf = PDFReader().load_data(file=pdf_path)
property_act_index = get_index(property_act_pdf, "bc strata property act")
property_engine = property_act_index.as_query_engine()

