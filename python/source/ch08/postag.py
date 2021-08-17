from nltk import pos_tag, word_tokenize
from yellowbrick.text.postag import PosTagVisualizer


pie = """
    In a small saucepan, combine sugar and eggs
    until well blended. Cook over low heat, stirring
    constantly, until mixture reaches 160° and coats
    the back of a metal spoon. Remove from the heat.
    Stir in chocolate and vanilla until smooth. Cool
    to lukewarm (90°), stirring occasionally. In a small
    bowl, cream butter until light and fluffy. Add cooled
    chocolate mixture; beat on high speed for 5 minutes
    or until light and fluffy. In another large bowl,
    beat cream until it begins to thicken. Add
    confectioners' sugar; beat until stiff peaks form.
    Fold into chocolate mixture. Pour into crust. Chill
    for at least 6 hours before serving. Garnish with
    whipped cream and chocolate curls if desired.
    """

tokens = word_tokenize(pie)
tagged = pos_tag(tokens)

visualizer = PosTagVisualizer()
visualizer.transform(tagged)

print(' '.join((visualizer.colorize(token, color)
                for color, token in visualizer.tagged)))
print('\n')


nursery_rhyme = """
    Baa, baa, black sheep,
    Have you any wool?
    Yes, sir, yes, sir,
    Three bags full;
    One for the master,
    And one for the dame,
    And one for the little boy
    Who lives down the lane.
    """


tokens = word_tokenize(nursery_rhyme)
tagged = pos_tag(tokens)

visualizer = PosTagVisualizer()
visualizer.transform(tagged)

print(' '.join((visualizer.colorize(token, color)
                for color, token in visualizer.tagged)))
print('\n')
