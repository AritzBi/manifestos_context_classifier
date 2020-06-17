
# -*- coding: utf-8 -*-
POLITICAL_PARTIES = ['UPN', 'CIUTADANS', 'PP', 'EHBILDU', 'PRC', 'CC', 'PARTIDO-RIOJANO', 'IU', 'PAR', 'PNV', 'PSOE', 'UPL', 'PCA']
BRITISH_POLITICAL_PARTIES = {"51320":"Labour-Party","51340":"Social-Democratic-Labour-Party","51421":"Liberal-Democrats","51620":"Conservative-Party","51621":"Ulster-Unionist-Party","51901": "The-Party-of-Wales", "51902": "Scottish-National-Party", "51903": "Democratic-Unionist-Party", "51951": "United-Kingdom-Independence-Party", "51210": "We-Ourselves","51110": "Green-Party-England-Wales" }
COUNTRY_PARTIES = {"spanish": ['33020', '33025', '33091', '33092', '33093', '33095', '33210', '33220', '33320', '33420', '33440', '33610', '33611', '33612', '33902', '33903', '33906', '33907', '33909', '33910'],
                   "danish": ['13001', '13229', '13230', '13320', '13330', '13410', '13420', '13520', '13620', '13720', '13951'],
                   "french": ['31021', '31110', '31220', '31230', '31240', '31320', '31421', '31425', '31430', '31624', '31626', '31630', '31631', '31720', '43902'],
                   "finnish": ['14110', '14223', '14320', '14520', '14620', '14810', '14820'],
                   "english": ['181210', '181310', '181410', '181411', '181420', '181910', '51110', '51210', '51320', '51340', '51421', '51620', '51621', '51901', '51902', '51903', '51951', '53021', '53110', '53230', '53231', '53240', '53250', '53320', '53321', '53420', '53520', '53620', '53951', '53981', '61320', '61620', '63110', '63320', '63410', '63620', '63621', '63622', '63710', '63810', '64110', '64320', '64420', '64421', '64422', '64620', '64621', '64901', '64902', '64951'],
                   "italian": ['32021', '32061', '32230', '32440', '32450', '32460', '32530', '32630', '32640', '32720', '32902', '32906', '32956', '43901'],
                   "german": ['32904', '41113', '41221', '41222', '41223', '41320', '41420', '41521', '41952', '41953', '42110', '42220', '42320', '42420', '42430', '42520', '42710', '42951', '43110', '43120', '43220', '43320', '43420', '43520', '43530', '43540', '43711', '43810', '43811']}

#7
DOMAIN_CLASSES = ['1','2','3','4','5','6','7']
#20 but we don't have enough data in three of them: 81, 91, 92,11. Therefore 16 differnt classes.
TERRITORIAL_CLASSES = ['10', '12', '20', '21', '22', '30', '31', '32', '80', '82', '90', '01', '02', '03', '09', '00']
#77 but we don't have enough data in three of them: 102, 103, 109. Therefore, 74 different classes.
SUBDOMAIN_CLASSES = ['101', '104', '105', '106', '107', '108', '110', '1017', '1027', '201', '202', '203', '204', '2024', '2025', '301', '302', '303', '304', '305','3012','3013','3014', '3031', '401', '402','403', '404', '405', '406', '407', '408', '409', '410', '411', '4111', '412', '413', '414', '415', '416', '501', '502', '503', '5032', '504', '5042','505', '5051','506', '5062', '507', '5071', '601', '6015', '6016', '6017', '602', '603', '604', '605', '6051', '606','607','608','701','702','703','704','705','7053','7054','706','000']
#609 but we don't have enough data in 240 of them. Therefore, 369 different classes.
COMPLETE_CLASSES = ['09_408', '30_605', '09_3012', '20_706', '20_704', '20_705', '20_702', '20_703', '30_408', '20_701', '30_406', '30_404', '30_405', '22_703', '10_416', '10_414', '10_411', '10_410', '02_502', '30_302', '30_305', '22_404', '30_7053', '00_410', '00_411', '30_608', '00_416', '30_705', '30_704', '30_706', '30_701', '03_607', '03_605', '20_410', '30_5062', '30_606', '01_504', '01_503', '01_502', '00_706', '00_701', '00_703', '20_409', '20_408', '09_105', '20_405', '20_404', '20_407', '20_406', '20_401', '20_403', '20_402', '09_503', '12_701', '09_502', '03_402', '03_403', '03_404', '03_405', '03_406', '10_5032', '03_408', '30_403', '00_000', '90_7053', '80_107', '80_104', '02_408', '80_108', '30_401', '20_101', '20_104', '20_105', '20_106', '20_107', '20_108', '30_202', '30_203', '30_204', '22_2024', '22_2025', '12_403', '22_204', '20_5051', '22_7053', '22_203', '09_3031', '90_501', '90_503', '32_4111', '22_405', '10_107', '30_603', '00_506', '00_504', '00_502', '00_503', '00_501', '82_408', '22_414', '03_108', '12_408', '09_2025', '09_2024', '01_301', '01_303', '30_501', '30_3031', '30_607', '30_5032', '10_305', '10_304', '10_303', '10_301', '00_605', '00_606', '80_701', '02_605', '80_703', '90_000', '30_2025', '30_2024', '22_701', '09_402', '09_403', '09_401', '80_410', '00_303', '03_000', '21_302', '22_403', '09_4111', '10_606', '09_606', '90_703', '30_506', '30_504', '30_503', '30_502', '22_410', '10_605', '20_2024', '20_2025', '09_705', '09_701', '80_416', '22_413', '20_607', '20_606', '20_605', '20_604', '10_3012', '20_601', '20_608', '09_703', '22_411', '12_506', '12_504', '12_503', '12_502', '12_501', '20_602', '09_501', '03_5032', '90_411', '09_605', '30_4111', '80_6051', '00_201', '20_415', '20_412', '20_6017', '20_413', '30_6015', '03_703', '30_000', '03_706', '03_705', '22_6015', '10_401', '10_402', '10_403', '10_404', '10_405', '10_408', '02_504', '03_606', '22_3014', '80_503', '90_408', '30_703', '09_201', '09_202', '09_203', '30_409', '32_401', '00_2024', '02_303', '20_501', '20_1017', '90_416', '03_501', '03_503', '03_502', '03_504', '03_506', '30_604', '30_413', '22_303', '20_5062', '30_201', '80_000', '20_204', '20_416', '20_000', '20_414', '90_410', '10_4111', '20_411', '80_506', '80_504', '03_1017', '80_501', '12_301', '30_414', '12_303', '30_410', '01_408', '22_502', '22_501', '22_506', '22_504', '31_301', '22_503', '03_4111', '90_107', '20_6015', '22_1017', '00_2025', '10_701', '20_302', '10_000', '20_303', '20_201', '20_3012', '20_3013', '20_3014', '30_402', '12_3012', '30_301', '30_7054', '30_303', '30_304', '30_416', '31_3031', '22_3031', '22_601', '09_506', '09_504', '22_605', '80_605', '00_107', '00_104', '80_2024', '30_6051', '20_305', '30_6017', '03_416', '03_107', '03_414', '03_411', '03_410', '20_301', '00_403', '00_402', '00_401', '20_6016', '20_603', '30_1017', '00_409', '00_408', '20_504', '20_505', '20_506', '20_507', '03_6051', '20_502', '20_503', '03_3031', '00_7053', '90_403', '30_601', '20_110', '12_605', '20_3031', '10_703', '09_107', '09_000', '09_106', '22_3012', '22_3013', '30_3013', '22_107', '20_4111', '80_201', '82_108', '80_203', '30_101', '80_202', '30_106', '30_104', '30_411', '20_304', '12_2025', '02_301', '22_108', '01_703', '10_501', '10_503', '10_502', '10_504', '10_506', '03_5042', '20_5032', '20_1027', '31_108', '20_7053', '90_701', '80_411', '22_301', '09_414', '09_416', '09_411', '09_410', '03_701', '00_202', '10_2025', '10_2024', '20_5071', '10_202', '30_107', '03_7053', '10_706', '00_3031', '30_108', '12_4111', '20_7054', '22_402', '80_408', '22_4111', '80_403', '20_6051', '03_2025', '03_2024', '03_3012', '80_414', '20_5042', '22_408', '09_301', '12_411', '09_303', '09_305', '09_304', '20_203', '20_202', '03_305', '03_304', '03_303', '03_301']
MINI_SUBDOMAIN_CLASSES = ['107', '108110', '201', '202', '203204', '301', '3012', '303', '3031', '305', '402', '403', '405', '408', '410', '411', '4111', '414', '416', '501', '502', '503', '504505', '506507', '601602', '605', '606', '701702', '703',  '704', '705', '706', '0' ]
BRAULIO_CLASSES = ['europe', 'welfare', 'economy', 'democracy_regeneration', 'immigration', 'territorial_debate',
                   'critics',
                   'rebaba']

BRAULIO_CLASSES_NO_REBABA = ['europe', 'welfare', 'economy', 'democracy_regeneration', 'immigration', 'territorial_debate',
                   'critics']
SUBBRAULIO_CLASSES = ['110', '108', '504', '506', '503', '413', '412', '406', '409', '404', '401', '414', '405', '4111', '403', '408', '402', '410', '411', '701', '702', '202', '203', '2024', '2025', '304', '5032', '5042', '5051', '5062', '5071', '6051', '607', '608', '301', '302', '3012', '3013', '3014', '601', '6015', '6016', '6017', '602', '7053', '305', '0']
#RANDOM_SEED = 0
#RMP_DATASET = "./datasets/rmp_dataset/2_tokenized/"
EMB_FOLDER = "./models/"
FOLDS_NUMBER = 5
UNWANTED_CHARACTERS = ['"', ',', '.', '\x80', '\x82', '\x89', '\x93', '\x99', '\x98', '\x9d', '\x9c', '\xa1', '\xa3', '"', '%', '$', '&', ')', '(', '+', '\xaa', '\xac', '/', '\xb2', '\xb4', '\xbb', '\xc3', '\xc2', '\xa0', '\xcc', '\xd2', '[', '\xdb', ']', '#', '`', '\xe2', '\xe5', '\xa6', '\xa9', '\xf7','-',';','!','?',':']
UNWANTED_CODE_CHARACTERS = ['"']

BRITISH_MANIFESTOS_SUBDOMAINS = ['605', '604', '607', '606', '601', '603', '602', '706', '701', '702', '703', '110', '406', '407', '0', '405', '404', '403', '402', '401', '506', '504', '505', '502', '503', '409', '501', '201', '203', '202', '204', '301', '302', '303', '304', '305', '608', '108', '109', '103', '101', '106', '107', '104', '105', '414', '416', '410', '411', '412', '413', '408']
BRITISH_MANIFESTOS_DOMAINS = ['0','1','2','3','4','5','6','7']
ALPHABET = 'abcdefghijklmnopqrstuvwxyz0123456789\''
GLOBAL_MANIFESTOS_DOMAINS = ['0','1','2','3','4','5','6','7']
GLOBAL_MANIFESTOS_SUBDOMAINS = ['0', '101', '102', '103', '104', '105', '106', '107', '108', '109', '110', '201',
                                '202', '203', '204', '301', '302', '303', '304', '305', '401', '402', '403', '404',
                                '405', '406', '407', '408', '409', '410', '411', '412', '413', '414', '415', '416',
                                '501', '502', '503', '504', '505', '506', '507', '601', '602', '603', '604', '605',
                                '606', '607', '608', '701', '702', '703', '704', '705', '706']
SUBSUB_TO_SUB = {'103.1': '103', '103.2': '103', '201.1': '201', '201.2': '201', '202.1': '202', '202.2': '202',
                 '202.3': '202', '202.4': '202', '305.1': '305','305.2': '305','305.3': '305','305.4': '305',
                 '305.5': '305', '305.6': '305', '416.1': '416', '416.2': '416', '601.1': '601', '601.2': '601',
                 '602.1': '602', '602.2': '602', '605.1': '605', '605.2': '605', '606.1': '606', '606.2': '606',
                 '607.1': '607', '607.2': '607', '607.3': '607', '608.1': '608', '608.2': '608', '608.3': '608',
                 '703.1': '703', '703.2': '703', '000': '0'}

BANNED_SUBS_FOR_HT = ['103', '204', '302', '408', '409', '416', '507', '608', '702']

TWITTER_TO_PARTY = {
	"SenTedCruz": "61620",
	"DorisMatsui": "61320",
	"Robert_Aderholt": "61620",
	"SteveDaines": "61620",
	"SenMikeLee": "61620",
	"marcorubio": "61620",
	"BobbyScott": "61320",
	"BillPascrell": "61320",
	"BettyMcCollum04": "61320",
	"BennieGThompson": "61320",
	"AustinScottGA08": "61620",
	"AppropsDems": "61320",
	"AOC": "61320",
	"RepAOC": "61320",
	"aguilarpete": "61320",
	"amyklobuchar": "61320",
	"RepBillFlores": "61620",
	"GrahamBlog": "61620",
	"RepDLamborn": "61620",
	"SenPatRoberts": "61620",
	"RepMcNerney": "61320",
	"RepKatiePorter": "61320",
	"RepJimmyPanetta": "61320",
	"RepBlaine": "61620",
	"GReschenthaler": "61620",
	"RepRichmond": "61320",
	"rosadelauro": "61320",
	"SenKamalaHarris": "61320",
	"RepVeasey": "61320",
	"RepTipton": "61620",
	"SenShelby": "61620",
	"RepTorresSmall": "61320",
	"RepMcGovern": "61320",
	"RepMichaelWaltz": "61620",
	"RonWyden": "61320",
	"MikeCrapo": "61620",
	"chelliepingree": "61320",
	"RepAnnaEshoo": "61320",
	"timkaine": "61320",
	"SenAlexander": "61620",
	"SteveKingIA": "61620",
	"SenJackReed": "61320",
	"SenatorCantwell": "61320",
	"RepLaHood": "61620",
	"RepPeteKing": "61620",
	"RepHuffman": "61320",
	"RepGregoryMeeks": "61320",
	"CongressmanRuiz": "61320",
	"RepAndyBiggsAZ": "61620",
	"USRepKeating": "61320",
	"RepGaramendi": "61320",
	"congbillposey": "61620",
	"TXRandy14": "61620",
	"KYComer": "61620", 
	"ABridgen": "51620",
	"ACunninghamMP": "51320",
	"AdamAfriyie": "51620",
	"AdrianBaileyMP": "51320",
	"AJonesMP": "51620",
	"AlanBrownSNP": "51902",
	"alancampbellmp": "51320",
	"AlanDuncanMP": "51620",
	"AlanMakMP": "51620",
	"alanwhiteheadmp": "51320",
	"AlbertoCostaMP": "51620",
	"AlbertOwenMP": "51320",
	"AlecShelbrooke": "51620",
	"ALewerMBE": "51620",
	"alexburghart": "51620",
	"AlexChalkChelt": "51620",
	"alexsobel": "51320", 
	"Alison_McGovern": "51320",
	"alisonthewliss": "51902",
	"AlistairBurtUK": "51620",
	"AlunCairns": "51620",
	"amandamilling": "51620",
	"AmberRuddHR": "51620",
	"andreajenkyns": "51620",
	"andrealeadsom": "51620",
	"Andrew4Pendle": "51620",
	"AndrewRosindell": "51620",
	"AndrewSelous": "51620",
	"AndyMcDonaldMP": "51320",
	"AngelaCrawley30": "51902",
	"angelaeagle": "51320",
	"AngelaRayner": "51320",
	"angelasmithmp": "51320",
	"AngusMacNeilSNP": "51902",
	"Anna_Soubry": "51620",
	"AnnaMcMorrin": "51320",
	"AnnClwyd": "51320",
	"anncoffey_mp": "51320",
	"AnnelieseDodds": "51320",
	"AnneMarieMorris": "51620",
	"AnneMilton": "51620",
	"annietrev": "51620",
	"ANorrisMP": "51320",
	"AWMurrison": "51620",
	"BambosMP": "51320",
    "RepJoshHarder": "61320",
    "BarryGardiner": "51320",
    "BarryMcElduff": "51210",
    "BarrySheerman": "51320",
    "BenMLake": "51901",
    "BenPBradshaw": "51320",
    "bernardjenkin": "51620",
    "Bill_Esterson": "51320",
    "BillCashMP": "51620",
    "BimAfolami": "51620",
    "BlaenauGwentMP": "51320",
    "BobBlackman": "51620",
    "BorisJohnson": "51620",
    "bphillipsonMP": "51320",
    "BrandonLewis": "51620",
    "BrendanOHaraSNP": "51902",
    "BrineMP": "51620",
    "BWallaceMP": "51620",
    "cajardineMP": "51421",
    "CarolineFlintMP": "51320",
    "CarolineLucas": "51110",
    "CatherineWest1": "51320",
    "CatMcKinnell": "51320",
    "CatSmithMP": "51320",
    "CGreenUK": "51620",
    "CharlieElphicke": "51620",
    "RepDonBeyer": "61320",
    "SenJohnBarrasso": "61620",
    "SenJohnHoeven": "61620",
    "SenJohnKennedy": "61620",
    "SenJohnThune": "61620",
    "SenJoniErnst": "61620",
    "SenKamalaHarris": "61320",
    "SenKevinCramer": "61620",
    "SenMarkey": "61320"
}
#https://www.charbase.com/2026-unicode-horizontal-ellipsis
#https://www.fileformat.info/info/unicode/char/2019/index.htm
#https://www.charbase.com/00ac-unicode-not-sign
#https://www.fileformat.info/info/unicode/char/25a0/index.htm
#https://www.fileformat.info/info/unicode/char/2022/index.htm
#http://www.codetable.net/hex/a9
#https://www.charbase.com/f02f-unicode-invalid-character
#http://www.codetable.net/hex/b1
#http://www.fileformat.info/info/unicode/char/b0/index.htm
#https://www.charbase.com/00b4-unicode-acute-accent
#https://www.charbase.com/00b7-unicode-middle-dot
#https://www.charbase.com/00bb-unicode-right-pointing-double-angle-quotation-mark
#https://www.charbase.com/f0b7-unicode-invalid-character
#http://www.codetable.net/hex/a0
#https://www.charbase.com/f0a7-unicode-invalid-character
#http://www.fileformat.info/info/unicode/char/F04F/index.htm
#https://www.fileformat.info/info/unicode/char/f050/index.htm
#https://www.fileformat.info/info/unicode/char/220e/index.htm
#http://www.fileformat.info/info/unicode/char/2DA/index.htm
#https://www.charbase.com/f0e0-unicode-invalid-character
#https://www.charbase.com/f020-unicode-invalid-character
#https://www.fileformat.info/info/unicode/char/0fffd/index.htm
#https://www.charbase.com/f0fc-unicode-invalid-character
#https://www.fileformat.info/info/unicode/char/0fffd/index.htm
#https://chars.suikawiki.org/string?s=%5CuF021
#https://www.charbase.com/f023-unicode-invalid-character
#https://www.fileformat.info/info/unicode/char/203a/index.htm
#http://www.fileformat.info/info/unicode/char/37e/index.htm
#https://www.fileformat.info/info/unicode/char/2009/index.htm
#http://www.codetable.net/hex/ab
#https://www.fileformat.info/info/unicode/char/2192/index.htm
#https://www.fileformat.info/info/unicode/char/2039/index.htm
#https://www.fileformat.info/info/unicode/char/201a/index.htm
#https://www.charbase.com/f0de-unicode-invalid-character
#http://www.fileformat.info/info/unicode/char/f0a0/index.htm
#https://www.fileformat.info/info/unicode/char/25CF/index.htm
#https://www.fileformat.info/info/unicode/char/25ba/index.htm
#https://www.charbase.com/009d-unicode-operating-system-command
SEPARATION_CHARS = [u'.', u'-', u'/', u'|', u'_', u'\\', u'"', u'(', u')', u',', u';', u':', u'[', u']', u'!', u'¡',
                    u'¿', u'?', u'=', u'&', u'º', u'ª', u'_', u'^', u'+', u'*', u'%', u'#', u'_', u'<', u'>', u'@',
                    u'~', u'`', u'}', u'{', u'\u2026',u'\u2019', u'\xac', u'\u25a0', u'\u201d', u'\u201c', u'\u201f',
                    u'\u201e', u'\xa7',
                             u'\u2022', u'\xa9', u'\uf02f', u'\xb1',  u'\xb0', u'\xb4', u'\xb7', u'\xbb', u'\uf0b7',
                             u'\xa0', u'\uf0a7', u'\uf0a7', u'\uf04f',u'\uf050', u'\u220e', u'\u02da', u'\uf0e0',
                    u'\uf020',u'\ufffd', u'\uf0fc', u'\uf021', u'\uf023', u'\u203a', u'\u037e', u'\u2009',u'\xab',
                    u'\u2192', u'\u2039', u'\u201a', u'\uf0de', u'\uf0a0', u'\u25cf', u'\u25ba', u'\x9d', u'\uf081',
                    u'\u2666', u'\u21d2', u'\uf02d']

KNOWN_CHARS  = {
    u'\u2013': u'-',
    u'\u2212': u'-',
    u'\u2014': u'-',
    u'\u2019': u"'", #REVISAR
    u'\u2018': u"'", #REVISAR,
    u'\xa7': u'',
    u'\u20ac': u' euros',
    u'\xa3': u' pounds',
    u'\xad': u'-',
    u'\u2011': u'-',
    u'\u2010': u'-',

}

UNWANTED_LANGUAGE_CHARACTERS = {'spanish': ["'", u'\xa8'], 'danish': [], 'english': [], 'finnish': [], 'french': [u'\xa8'], 'german': [], 'italian': []}

# Convert weird characters to its basic form
KNOWN_CHARS_PORDUNA = {
    u'\u2010': u'-',
    u'\u2013': u'-',
    u'\u2019': u"'",
    u'\u2018': u"'",
    u'\u201d': u'"',
    u'\u201c': u'"',
    u'\u2026': u'...',
    u'\u20ac': u' euros ',
    u'\xb4' : u"'",
    u'\xa1': u'¡',
    u'\xa3': u'£',
    u'\xa4': u' euros ',
    u'\ufb01': u'fi',
    u'\ufb02': u'fl',
    u'\u2011': u'-',
    u'\u2212': u'-',
    u'\u2014': u'-',
    u'\u2015': u'-',
    u'\u201f': u'"',
    u'\u201e': u'"',
    u'\u2122': u'™',
    u'\xa8': u'"',
    u'\xab': u'"',
    u'\xbb': u'"',
    u'\xb0': u'º',
    u'\xb3': u'³',
    u'\xf2': u'ó',
    u'\xbd': u' media ',
    u'\xd6': u'Ó',
    u'\xe0': u'à',
    u'\xe2': u'â',
    u'\u2264': u'≤',
    u'\xe8': u'è',
    u'\xef': u'ï',
    u'\xbc': u'%',
    u'\u2022': u'', # bullet point
    u'\u203a': u'', # ›
    u'\uf08a': u'', # 
    u'\u25a0': u'', # ■
    u'\uf0a7': u'', # 
    u'\xad': u'', # (empty)
    u'\xac': u'', # ¬
    u'\uf02f': u'', #
    u'\xae': u'', # (copyright symbol used as bullet point)
    u'\xb7': u'', # bullet point
    u'\u223c': u'', # ~ used as bullet point
    u'\uf020': u'', # small .
    u'\xa0': u'',
    u'\xa7': u'', # §
    u'\u2192': u'', # →'
    u'\u0e4f': u'', # ๏
    u'\xd4': u'', # Ô as bullet point
    u'\uf0d8': u'', # 
    u'\u2666': u'', # ♦
    u'\u0192': u'', # ƒ
    u'\uf076': u'', # 
    u'\uf095': u'', # 
    u'\uf0fc': u'', # ,
    u'\r\x07': u'',
}

