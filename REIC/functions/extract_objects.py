import stanza
from stanza.pipeline.core import DownloadMethod
nlp = stanza.Pipeline('en', download_method=DownloadMethod.REUSE_RESOURCES, processors='tokenize,mwt,pos')


def get_objects_in_caption(caption):
    # get noun phrase from a caption
    caption = caption.replace('<unk>', 'unk')
    doc = nlp(caption)
    words = doc.sentences[0].words
    nn_list = []
    index_list = []

    last_noun = False
    caption_length = len(words)
    for i in range(caption_length):
        tag = words[i].xpos
        # noun before 'of' is ignored
        if i + 1 != caption_length and words[i + 1].text == 'of' or words[i].text == 'unk':
            continue
        if tag in ['NN', 'NNS']:
            if last_noun:
                # if last word is noun, then they are a phrase
                p_noun = nn_list.pop()
                index = index_list.pop()
                nn_list.append(f'{p_noun}_{words[i].text}')
                index_list.append(f'{index}_{i}')
            else:
                nn_list.append(words[i].text)
                index_list.append(i)
            last_noun = True
        else:
            last_noun = False
    return nn_list, index_list


def get_objects_in_followup_caption(caption):
    # get noun phrase from a caption
    caption = caption.replace('<unk>', 'unk')
    doc = nlp(caption)
    words = doc.sentences[0].words
    nn_list = []
    index_list = []

    last_noun = False
    caption_length = len(words)
    for i in range(caption_length):
        tag = words[i].xpos
        if words[i].text == 'unk':
            continue
        if tag in ['NN', 'NNS']:
            if last_noun:
                # if last word is noun, then they are a phrase
                p_noun = nn_list.pop()
                index = index_list.pop()
                nn_list.append(f'{p_noun}_{words[i].text}')
                index_list.append(f'{index}_{i}')
            else:
                nn_list.append(words[i].text)
                index_list.append(i)
            last_noun = True
        else:
            last_noun = False
    return nn_list, index_list


def get_noun(caption):
    # get noun from a caption
    caption = caption.replace('<unk>', 'unk')
    doc = nlp(caption)
    words = doc.sentences[0].words
    nn_list = []
    index_list = []
    caption_length = len(words)
    for i in range(caption_length):
        tag = words[i].xpos
        # noun before 'of' is ignored
        if i + 1 != caption_length and words[i + 1].text == 'of' or words[i].text == 'unk':
            continue
        if tag in ['NN', 'NNS']:
            nn_list.append(words[i].text)
            index_list.append(i)
    return nn_list, index_list