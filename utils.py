import editdistance
import re


# defines global regex for tagged noises
re_tagged_noises = re.compile(r"[\[<][A-Za-z ]*[\]>]")

# defines global regex to remove these nsns
non_silence_noises = ["noise", "um", "ah", "er", "umm", "uh", "mm", "mn", "mhm", "mnh", "<START>", "<END>"]
re_non_silence_noises = re.compile(r"\b({})\b".format("|".join(non_silence_noises)))


def remove_non_silence_noises(input_text):
    """
      Removes non_silence noises from a transcript
    """
    return re.sub(re_non_silence_noises, '', input_text)


def wer(ref, hyp, remove_nsns=False):
    """
      Calculate word error rate between two string or time_aligned_text objects
      >>> wer("this is a cat", "this is a dog")
      25.0
    """
    # remove tagged noises
    # ref = re.sub(re_tagged_noises, ' ', ref)
    # hyp = re.sub(re_tagged_noises, ' ', hyp)
    ref = re.sub('^<START>|<EOS>$', '', ref)
    hyp = re.sub('^<START>|<EOS>$', '', hyp)

    # optionally, remove non silence noises
    if remove_nsns:
        ref = remove_non_silence_noises(ref)
        hyp = remove_non_silence_noises(hyp)

    # clean punctuation, etc.
    # ref = clean_up(ref)
    # hyp = clean_up(hyp)

    # calculate WER
    return editdistance.eval(ref.split(' '), hyp.split(' ')), len(ref.split(' '))


def cer(ref, hyp, remove_nsns=False):
    """
      Calculate character error rate between two strings or time_aligned_text objects
      >>> cer("this cat", "this bad")
      25.0
    """
    ref = re.sub('^<START>|<EOS>$', '', ref)
    hyp = re.sub('^<START>|<EOS>$', '', hyp)

    if remove_nsns:
        ref = remove_non_silence_noises(ref)
        hyp = remove_non_silence_noises(hyp)

    # ref = clean_up(ref)
    # hyp = clean_up(hyp)

    # calculate per line CER
    return editdistance.eval(ref, hyp), len(ref)


def clean_up(input_line):
    """
      Apply all text cleaning operations to input line
      >>> clean_up("his license plate is a. c, f seven...five ! zero")
      'his license plate is a c f seven five zero'
      >>> clean_up("Q2")
      'q two'
      >>> clean_up("from our website at www.take2games.com.")
      'from our website at www take two games dot com'
      >>> clean_up("NBA 2K18")
      'n b a two k eighteen'
      >>> clean_up("launched WWE 2K 18")
      'launched w w e two k eighteen'
      >>> clean_up("released L.A. Noire, the The VR Case Files for the HTC VIVE system")
      'released l a noire the the v r case files for the h t c v i v e system'
      >>> clean_up("Total net bookings were $654 million,")
      'total net bookings were six hundred and fifty four million dollars'
      >>> clean_up("net booking which grew 6% to $380 million.")
      'net booking which grew six percent to three hundred and eighty million dollars'
      >>> clean_up("to $25 dollars or $0.21 per share price.")
      'to twenty five dollars dollars or zero dollars and twenty one cents per share price'
      >>> clean_up("year-over-year")
      'year over year'
      >>> clean_up("HTC VIVE")
      'h t c v i v e'
      >>> clean_up("you can reach me at 1-(317)-222-2222 or fax me at 555-555-5555")
      'you can reach me at one three one seven two two two two two two two or fax me at five five five five five five five five five five'
    """
    for char_to_replace in ",*&!?":
        input_line = input_line.replace(char_to_replace, ' ')

    # We do not need the right below for I-Keyboard
    # for pat in rematch:
    #     input_line = re.sub(rematch[pat][0], rematch[pat][1], input_line)

    for char_to_replace in ",.-":
        input_line = input_line.replace(char_to_replace, ' ')

    input_line = input_line.encode().decode('utf-8').lower()

    # check for double spacing
    while "  " in input_line:
        input_line = input_line.replace("  ", " ")
    return input_line.strip()


def batch_error(batch_ref, batch_hyp, translator, lengths, measure):
    total_err, total_len = 0, 0
    for i in range(len(batch_ref)):
        ref = translator.ids_to_string(batch_ref[i], lengths[i])
        hyp = translator.ids_to_string(batch_hyp[i], lengths[i])
        err, length = measure(ref, hyp)
        total_err += err
        total_len += length
    return total_err, total_len


def batch_cer(batch_ref, batch_hyp, translator, lengths):
    return batch_error(batch_ref, batch_hyp, translator, lengths, cer)


def batch_wer(batch_ref, batch_hyp, translator, lengths):
    return batch_error(batch_ref, batch_hyp, translator, lengths, wer)
