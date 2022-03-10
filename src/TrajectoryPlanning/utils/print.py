last_text_length = 0

def put(text: str, same_line: bool = False):
    global last_text_length
    text_length = len(text)
    space_count = max(0, last_text_length - text_length)
    text += ' ' * space_count

    if same_line:
        last_text_length = text_length
        print(text, end='\r')
    else:
        last_text_length = 0
        print(text)