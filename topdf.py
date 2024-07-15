from pyhpo import Ontology
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import os

class HPOManager:
    def __init__(self):
        # Load the HPO ontology
        self.ontology = Ontology()

    def get_term_by_name(self, term_name):
        # Retrieve a term by its name
        # term_name='scoliosis'
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
        self.data_directory = os.path.join(self.script_directory, r'C:\Users\simran kohi\locak\data')
        os.makedirs(self.data_directory, exist_ok=True)

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
        return file_path  # Return the file path of the generated PDF


# # Usage example
if __name__ == "__main__":
    # Initialize HPOManager
    hpo_manager = HPOManager()

    # Retrieve the term
    term_name = 'Scoliosis'
    term = hpo_manager.get_term_by_name(term_name)

    # Check if the term exists
    if term:
        # Initialize PDFGenerator
        pdf_generator = PDFGenerator()

        # Specify the output file name
        output_file = "hpo_term_info.pdf"

        # Generate the PDF with the term information
        pdf_generator.generate_pdf(output_file, term)
    else:
        print("Term not found.")
