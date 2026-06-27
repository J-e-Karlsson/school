import spacy

nlp = spacy.load("en_core_web_sm")

text = [
"Also if any man bi cause of seruyce or other leueful comaundement approched a lorde to which lorde he Nichol dradde his falshede to be knowe to anon was apeched that he was false to the conseille of the Citee & so to the kyng",
"For thy graciouse lordes lyke it to yow to take hede in what manere & where owre lige lordes power hath ben mysused by the forsaid Nichol & his vpberers for sithen thise wronges bifore saide han ben vsed as accidental or comune braunches outward it sheweth wel the rote of hem is a ragged subiect or stok inward that is the forsaid Brere or brembre the whiche comune wronge vses & many other if it lyke to yow mowe be shewed & wel knowen bi an indifferent Juge and Mair of owre Citee the which wyth yowre ryghtful lordeship ygraunted for moost pryncipal remedye as goddes lawe & al resoun wole that no domesman stonde togidre Juge & partye wronges sholle more openlich be knowe & trouth dor apere",
"Wryten at your forsayd Cite of london the xx day of Decembre"
]

norm = [
"Be your awne Jayn Stonor",
"Also me thynk thay sshuld nat be so wery of you that dyd so gret labour and diligence to have you and wher as ye thynk I sshuld be unkynd to you uerrely that am I nat for and ye be as I left you as I trust uerrely that ye be I am and wyll be to you as a moder sshuld be and if so be thay be wery of you ye sshall cum to me and ye wille your selfe so that my housbond or I may have writing fro the quene with her awn hand and ells he nor I neyther dar nor wyll take upon us to reseyve you seing the quenys displesyr afore for myn housbond seyth he hath nat wyllingly disobeyd her comaundment here afore nor he wyll nat begynne now",
"And for so mekyll as for shortnesse off tyme he m3yt noght utter till his holynesse his entent at that tyme he besoght his holynesse to contynew the said commyssyon till the next Consistory and the cumming of the King of France embassiatore3 the wilk shall entre Rome the setterday nex after the day affor reherset wheruppon the Pope was ryght wele content and so degreet",
"I besek your gud fadirhod to exhort our brether to pray hertly for the cause of Coldingham the wylk was proposet in Consistorio publico at cumming off embassiatore3 off the king of Portugaly and for shortnesse of tyme the pope continewit it to the next consistory wher shall do ther obedience the embassiatore3 off the king off France and your ryghte3 utterly decidit",
"London",
"Ano"
]

orig = [
"Be your awne Jayn Stonor",
"Also me thynk þay sshuld nat be so wery of yow þat dyd so gret labour and diligence to have yow and wher as ye thynk I sshuld be unkynde to yow verrely þat am I nat for and ye be as I left yow as I trust verrely þat ye be I am and wyll be to yow as a moder sshuld be and if so be þay be wery of yow ye sshall cum to me and ye wille your selfe so þat my housbond or I may have writyng fro þe quene with her awn hand and ells he nor I neyther dar nor wyll take upon us to reseyve yow seyng þe quenys displesyr afore for myn housbond seyth he hath nat wyllyngly disobeyde her comaundment here afore nor he wyll nat begynne nowe",
"And for so mekyll as for shortnesse off tyme he m3yt noght utter till his holynesse his entent att that tyme he besoght his holynesse to contynew the said commyssyon till the next Consistory and the cummyng of the Kyng of France embassiatore3 the wilk shall entre Rome the setterday nex after the day affor rehersett wheruppon the Pope was ryght wele content and so degreet",
"I besek your gud fadirhod to exhort our brether to pray hertly for the cause of Coldyngham the wylk was proposet in Consistorio publico att cummyng off embassiatore3 off the kyng of Portugaly and for shortnesse of tyme the pope continewit itt to the next consistory wher schall do ther obedience the embassiatore3 off the kyng off France and your ryghte3 utterly deciditt",
"London",
"Ano"
]


for sentence in text:
    doc = nlp(sentence)
    print("SENTENCE BREAK\n")
    for token in doc:
        print(token.text, token.pos_, "\n")

print("------------------------------------------------------------------------------------------------------")

for sentence in norm:
    doc = nlp(sentence)
    print("SENTENCE BREAK\n")
    for token in doc:
        print(token.text, token.pos_, "\n")

print("------------------------------------------------------------------------------------------------------")

for sentence in orig:
    doc = nlp(sentence)
    print("SENTENCE BREAK\n")
    for token in doc:
        print(token.text, token.pos_, "\n")