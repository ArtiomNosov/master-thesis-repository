# Word DOCX and python-docx reference for the converter

Sources:

- python-docx 1.2.0 documentation, https://python-docx.readthedocs.io/en/latest/
- Working with Styles, https://python-docx.readthedocs.io/en/latest/user/styles-using.html
- Working with Text, https://python-docx.readthedocs.io/en/latest/user/text.html
- Working with Sections, https://python-docx.readthedocs.io/en/latest/user/sections.html
- Working with Tables, https://python-docx.readthedocs.io/en/latest/user/tables.html

## DOCX concepts used

- A Word document is built from paragraphs, runs, tables and sections.
- Section properties define page size, orientation, margins and header/footer distances.
- Paragraph styles define reusable text formatting: font, size, bold, indentation, spacing and pagination behavior.
- Runs carry inline formatting such as bold, italic and inline code font.
- Tables are block-level objects; each cell contains paragraphs.
- Page numbers are represented by Word field XML (`PAGE`) in the footer.

## python-docx APIs used

- `Document()` creates a DOCX document.
- `document.sections[0]` configures page size and margins.
- `document.styles` configures paragraph styles.
- `document.add_paragraph()` writes paragraphs and headings.
- `document.add_table()` writes Markdown tables.
- `document.add_picture()` writes local Markdown images when paths exist.
- `docx.shared.Cm`, `Mm`, `Pt` set Word measurements.
- `WD_ALIGN_PARAGRAPH` sets left, center, right and justify alignment.
- Low-level XML is used only for features not exposed as high-level helpers, including Cyrillic font bindings and footer page-number fields.

## Formatting checks

The checker reads the generated `.docx` file with `python-docx` and, where necessary, direct WordprocessingML XML. It verifies structural properties such as margins, styles, paragraph spacing, first-line indentation, caption style alignment and footer page-number fields. It does not use visual inspection.
