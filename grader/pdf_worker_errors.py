"""Static error codes shared by the isolated PDF worker and its helpers."""


class PdfPreparationError(Exception):
    def __init__(self, code):
        self.code = code
