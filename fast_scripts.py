import json, codecs
from constants import BRAULIO_CLASSES_NO_REBABA, GLOBAL_MANIFESTOS_SUBDOMAINS, DOMAIN_CLASSES,GLOBAL_MANIFESTOS_SUBDOMAINS
languages = ['german', 'danish', 'spanish', 'italian', 'english', 'finnish', 'french']
dataset = "./datasets/multilingual/"


def compute_party_lists():
    for language in languages:
        path = dataset + language + "/2_tokenized/" + language + ".json"
        print "Loading...", path
        data = json.load(open(path, 'r'))
        parties = []
        for row in data:
            if row['party_code'] not in parties:
                parties.append(row['party_code'])
        print "The parties of language " + language + " are: "
        print parties


def compute_class_distribution():
    path = "./datasets/annotated_tweets/2_tokenized/all_annotated_tweets_braulio_no_rebaba.json"
    data = json.load(open(path, 'r'))
    print "Total of tweets: %s", len(data)
    distribution = {}
    for e in BRAULIO_CLASSES_NO_REBABA:
        distribution[e] = 0
    for d in data:
        distribution[d['braulio']] += 1
    print distribution

    for key in distribution:
        print key + ": " + str(distribution[key]/float(len(data))*100) + "%"
#compute_class_distribution()

def remove_zero_from_datasets():
    for language in languages:
        path = dataset + language + "/2_tokenized/"
        source_file = path + language + ".json"
        target_file = path + language + "nozero.json"
        source = json.load(open(source_file, 'r'))
        target = []
        for s in source:
            s['domain'] = s['manifestos_domain']
            if s['manifestos_domain'] != '0':
                target.append(s)
        json.dump(target, codecs.open(target_file, 'w', encoding='utf-8'), indent=4)
        print language
        print "Source len ", len(source)
        print "Target len", len(target)
#remove_zero_from_datasets()

def compute_datasets_statistics():
    for language in languages:
        path = dataset + language + "/2_tokenized/"
        source_file = path + language + ".json"
        source = json.load(open(source_file, 'r'))
        number_senteces = len(source)
        number_programs = []
        for s in source:
            number_programs.append(s['file'])
        len_programs = len(set(number_programs))
        print language + " - " + str(number_senteces) + " - " + str(len_programs)
#compute_datasets_statistics()
#print str(len(GLOBAL_MANIFESTOS_SUBDOMAINS))

def compute_datasets_statistics():
    for language in languages:
        path = dataset + language + "/2_tokenized/"
        source_file = path + language + "nozero.json"
        source = json.load(open(source_file, 'r'))
        distribution = {}
        for e in DOMAIN_CLASSES:
            distribution[e] = 0
        for d in source:
            distribution[d['domain']] += 1
        print distribution
        print language
        for key in distribution:
            print key + ": " + str(distribution[key] / float(len(source)) * 100) + "%"

#compute_datasets_statistics()

def compute_subdomain_statistics():
    for language in languages:
        path = dataset + language + "/2_tokenized/"
        source_file = path + language + ".json"
        source = json.load(open(source_file, 'r'))
        distribution = {}
        for e in GLOBAL_MANIFESTOS_SUBDOMAINS:
            distribution[e] = 0
        for d in source:
            distribution[d['manifestos_subdomain']] += 1
        print distribution
        print language
        for key in distribution:
            print key + ": " + str(distribution[key] / float(len(source)) * 100) + "%"

#compute_subdomain_statistics()

def generate_placeholder(length):
    placeholder = []
    for i in range(length):
        placeholder.append([])
    return placeholder