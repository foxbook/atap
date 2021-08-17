import json
from nltk.stem.snowball import SnowballStemmer

CONVERSION_PATH = "conversions.json"

with open(CONVERSION_PATH, 'r') as f:
    conv_dict = json.load(f)

def conversion(source, dest):
    """
    :param source: the unit of measure you have
    :param dest: the unit of measure need to convert to
    :return:
    """
    stemmer = SnowballStemmer('english')
    source = stemmer.stem(source)
    dest = stemmer.stem(dest)

    try:
       units = conv_dict.get(source).get('Units')[
          conv_dict.get(source).get('Destination').index(dest)
       ]
    except:
       units = None

    return units, source, dest

if __name__ == "__main__":
    units, source, dest = conversion('teaspoons', 'tablespoons')
    if units == None:
       print("Sorry, I cannot do that conversion. Try again.")
    else:
        import inflect
        engine = inflect.engine()
        print("There are {} {} in a {}.".format(
            units, engine.plural(source), dest)
        )