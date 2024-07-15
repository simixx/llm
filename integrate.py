from pyhpo import Ontology
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import os
from PyPDF2 import PdfWriter, PdfReader
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.output_parsers import RetryOutputParser

# Load environment variables
import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = "gpt-3.5-turbo"
#MODEL = "mixtral:8x7b"
MODEL = "llama2"


class HPOManager:
    def __init__(self):
        # Load the HPO ontology
        self.ontology = Ontology()

    def get_term_by_name(self, term_name):
        # Retrieve a term by its name
        term = self.ontology.get_hpo_object(term_name)
        return term

    def get_term_by_id(self, term_id):
        # Fetch the HPOTerm object for the specific term ID
        term = self.ontology.get_hpo_object(term_id)
        return term

class PDFGenerator:
    def __init__(self):
        # Get the directory where the script is located
        self.script_directory = os.path.dirname(os.path.abspath(__file__))
        # Create a data folder in the same directory as the script
        self.data_directory = os.path.join(self.script_directory, 'data')
        os.makedirs(self.data_directory, exist_ok=True)
        print(f"Data directory created at: {self.data_directory}")

    def generate_pdf(self, filename, term):
        # Create the full path to the file in the data folder
        file_path = os.path.join(self.data_directory, filename)
        
        # Create a PDF document
        doc = SimpleDocTemplate(file_path, pagesize=letter)
        styles = getSampleStyleSheet()

        # Create a list to hold the flowable elements of the PDF
        elements = []

        # Add a title
        title = Paragraph("HPO Term Information", styles['Title'])
        elements.append(title)

        # Add a spacer
        elements.append(Spacer(1, 12))

        # Add attributes of the HPOTerm object to the PDF
        for attribute, value in term.__dict__.items():
            if value is not None:
                if isinstance(value, set):
                    if attribute == 'genes':
                        gene_info = []
                        for gene in value:
                            gene_info.append("Gene ID: {}, Gene Name: {}".format(gene.id, gene.name))
                        value = ", ".join(gene_info)
                    else:
                        value = ', '.join(str(item) for item in value)
                elif isinstance(value, bool):
                    value = 'Yes' if value else 'No'
                attribute_text = f"<b>{attribute.replace('_', ' ').title()}:</b> {value}"
                elements.append(Paragraph(attribute_text, styles['Normal']))

        # Build the PDF document
        doc.build(elements)

        print("PDF generated successfully:", file_path)
        return file_path

def split_pdf_pages(input_pdf_path, output_dir):
    pdf_reader = PdfReader(open(input_pdf_path, 'rb'))
    num_pages = len(pdf_reader.pages)
    print(f"Number of pages in PDF: {num_pages}")

    for i in range(num_pages):
        pdf_writer = PdfWriter()
        pdf_writer.add_page(pdf_reader.pages[i])

        output_pdf_path = os.path.join(output_dir, f"page_{i + 1}.pdf")
        with open(output_pdf_path, 'wb') as output_pdf:
            pdf_writer.write(output_pdf)
        print(f"Page {i + 1} saved as {output_pdf_path}")

    return num_pages

def load_and_prepare_rag_model(split_pdf_dir):
    documents = []
    for filename in os.listdir(split_pdf_dir):
        if filename.endswith('.pdf'):
            loader = PyPDFLoader(os.path.join(split_pdf_dir, filename))
            documents.extend(loader.load_and_split())
            print(f"Loaded and split {filename}")

    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectorstore = DocArrayInMemorySearch.from_documents(documents, embedding=embeddings)
    
    template = """
    Answer the question based on the context below. If you can't 
    answer the question, reply "I don't know".

    Context: {context}

    Question: {question}
    """

    prompt = PromptTemplate.from_template(template)
    llm = OpenAI(model="text-davinci-003", openai_api_key=OPENAI_API_KEY)

    chain = ConversationalRetrievalChain(
        retriever=vectorstore.as_retriever(),
        llm=llm,
        prompt=prompt,
        output_parser=StrOutputParser()
    )
    print("RAG model loaded and prepared.")
    return chain

if __name__ == "__main__":
    # Initialize HPOManager
    hpo_manager = HPOManager()

    # Retrieve the term
    term_name = 'Scoliosis'
    term = hpo_manager.get_term_by_name(term_name)
    print(f"Term retrieved: {term_name}")

    # Check if the term exists
    if term:
        # Initialize PDFGenerator
        pdf_generator = PDFGenerator()

        # Specify the output file name
        output_file = "hpo_term_info.pdf"

        # Generate the PDF with the term information
        pdf_path = pdf_generator.generate_pdf(output_file, term)

        # Split the PDF into pages
        split_pdf_dir = os.path.join(pdf_generator.data_directory, 'split_pages')
        os.makedirs(split_pdf_dir, exist_ok=True)
        split_pdf_pages(pdf_path, split_pdf_dir)

        # Load and prepare RAG model
        chain = load_and_prepare_rag_model(split_pdf_dir)

        # Test the RAG model
        question = "What are the symptoms of Scoliosis?"
        result = chain.invoke({"context": "Some context from the PDF", "question": question})
        print(f"RAG model result: {result}")
    else:
        print("Term not found.")
