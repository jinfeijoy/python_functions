import requests
import pdfplumber
from io import BytesIO
from urllib.request import urlopen
import io

def extract_text_from_texturl(url_link, filename):
    file = open(filename,"w")
    url = urlopen(url_link)
    for line in url:
        file.write(str(line) + '\n')
    file.close()

def extract_text_from_pdfurl(url_link, filename):
    response = requests.get(url_link)
    my_raw_data = response.content
    with BytesIO(my_raw_data) as data:
        with pdfplumber.open(data) as pdf:
            pages = pdf.pages
            outtext = ''
            for i,pg in enumerate(pages):
                tbl = pages[i].extract_text()
                outtext+=(f'{tbl}')
    with io.open(filename, 'w', encoding="utf-8") as text_file:
        text_file.write(outtext)