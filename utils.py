import PyPDF2
import json

def fix_unicode_characters(json_path: str):
    """
    Read JSON file, replace Unicode placeholders with actual characters, and write back
    
    Args:
        json_path: Path to the JSON file
    """
    try:
        # Read the JSON file
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Function to recursively process dictionary values
        def process_dict(d):
            for key, value in d.items():
                if isinstance(value, str):
                    # Replace the Unicode placeholder with actual character
                    d[key] = value.replace('\u2014', '—')
                elif isinstance(value, list):
                    # Process lists
                    for item in value:
                        if isinstance(item, dict):
                            process_dict(item)
                elif isinstance(value, dict):
                    # Process nested dictionaries
                    process_dict(value)
            return d
        
        # Process the data
        processed_data = process_dict(data)
        
        # Write back to file with proper formatting
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, indent=4, ensure_ascii=False)
            
        print(f"Successfully processed and saved {json_path}")
        
    except Exception as e:
        print(f"Error processing file: {str(e)}")

def save_pdf_pages(pdf_path, page_from, page_to, output_path=None):
    """
    Extract a range of pages from a PDF and save them as a new PDF file.
    
    Parameters:
    pdf_path (str): Path to the input PDF file
    page_from (int): Starting page number (1-based indexing like in PDF viewers)
    page_to (int): Ending page number (1-based indexing like in PDF viewers)
    output_path (str, optional): Path for the output PDF file. If not provided, 
                               it will use the original filename with "_extracted" suffix
    
    Returns:
    str: Path to the saved PDF file
    """
    # If output path is not specified, create one
    if output_path is None:
        # Split the path to get directory and filename
        parts = pdf_path.rsplit('.', 1)
        output_path = parts[0] + "_extracted.pdf"
    
    # Open the input PDF file
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        total_pages = len(pdf_reader.pages)
        
        # Validate page range
        if page_from < 1:
            page_from = 1
            print(f"Warning: Starting page adjusted to 1")
            
        if page_to > total_pages:
            page_to = total_pages
            print(f"Warning: Ending page adjusted to {total_pages}")
            
        if page_from > page_to:
            print(f"Error: Starting page ({page_from}) cannot be greater than ending page ({page_to})")
            return None
        
        # Create a PDF writer object
        pdf_writer = PyPDF2.PdfWriter()
        
        # Add the pages in the specified range to the writer
        for page_num in range(page_from, page_to + 1):
            # Convert from 1-based to 0-based indexing
            idx = page_num - 1
            
            # Add the page to the writer
            pdf_writer.add_page(pdf_reader.pages[idx])
        
        # Write the output PDF file
        with open(output_path, 'wb') as output_file:
            pdf_writer.write(output_file)
    
    pages_extracted = page_to - page_from + 1
    print(f"Extracted {pages_extracted} pages ({page_from} to {page_to}) to {output_path}")
    return output_path