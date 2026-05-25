# Markdown/CommonMark reference for the converter

Sources:

- CommonMark Spec 0.31.2, https://spec.commonmark.org/0.31.2/
- `markdown-it-py` parser behavior is used through the CommonMark-compatible token stream.

## Supported input scope

The converter accepts UTF-8 Markdown and supports the Markdown constructs that can be represented in a `.docx` document without semantic guessing:

- ATX headings: `#`, `##`, `###` and deeper levels.
- Paragraphs.
- Strong emphasis: `**text**`.
- Emphasis: `*text*`.
- Inline code: `` `code` ``.
- Fenced code blocks: triple backticks.
- Ordered and unordered lists.
- Pipe tables when the parser emits table tokens.
- Thematic breaks as page breaks.
- Markdown images with local file paths when the image file exists.
- Plain links: link text is preserved; the current converter does not create Word hyperlink relationships.

## Parsing rules relevant to this tool

- Markdown is parsed in two layers: block structure first, inline structure second.
- ATX headings are defined by one to six leading `#` characters followed by a space or tab.
- A blank line separates paragraphs.
- Tabs behave as four-space tab stops where indentation defines block structure.
- Backslash escaping works for ASCII punctuation outside code spans and code blocks.
- Fenced code blocks preserve code text and are formatted separately in DOCX.

## Thesis-draft compatibility rule

The current thesis draft contains some headings as full bold paragraphs rather than ATX headings. The converter has an explicit compatibility option:

- a full strong paragraph matching `Введение`, `Заключение`, `Список ...` or a numbered top-level heading can be promoted to a Word heading;
- shorter full strong paragraphs can be promoted to a lower-level heading.

This is a Markdown-to-DOCX formatting rule, not a content rule.
