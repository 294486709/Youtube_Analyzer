import re


def split_sentence(filename):
    '''
    the function is to convert the article into a list and
    # split it into sentence
    # '''

    with open('./articles/%s.txt' %filename, 'r', encoding='utf-8') as f:
        line = f.read().strip()
        line = re.sub(u'\\(.*?\\)', '', line)
        line = re.sub(r'[\n]', '', line)
        line = re.sub(u'\\\\', '', line)
        line = line.replace('\\', '')
        linestr = re.split(r'[.?!;]', line)
        linestr.pop()
    return linestr


# if __name__ == '__main__':
#     filename = '1'
#     split_sentence(filename)