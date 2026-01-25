# ************************************************************************************************************************************************************************************
# Imports
# ************************************************************************************************************************************************************************************

import numba
import string
import numpy as np
import pandas as pd
from scipy import stats
from collections import Counter, defaultdict

from typing import Any, Sequence, List, Optional, Callable

from pyutilz.strings import compute_entropy_stats, naive_entropy_rate

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# LOGGING
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

from os import path
import logging, logging.config

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# load the logging configuration
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# log_file_path = path.join(path.dirname(path.abspath("__file__")), 'logging.ini')
# logging.config.fileConfig(log_file_path, disable_existing_loggers=False)
# logger.setLevel(logging.INFO)

# ************************************************************************************************************************************************************************************
# Inits
# ************************************************************************************************************************************************************************************

nlp = None
spellers, langs = None, None
punctuation = string.punctuation
domain_suffixes = None
nlp_stopwords = set()
VOWELS = set(["a", "e", "i", "o", "u", "а", "у", "о", "и", "э", "ы", "я", "ю", "е", "ё"])  # latin  # cyrillic

# ************************************************************************************************************************************************************************************
# Numerical features
# ************************************************************************************************************************************************************************************

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Оптимизированные создавалки аггрегирующих характеристик
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


def numDifferentFractions(r):
    return len(np.unique(np.round_(np.modf(r.astype(float))[0], 5)))


def numUnique(x):
    return np.unique(x.values).size


def percentile10(x):
    return np.percentile(x, 10)


def percentile50(x):
    return np.percentile(x, 50)


def percentile90(x):
    return np.percentile(x, 90)


def minGap(x):
    try:
        y = np.sort(x)
        return np.nanmin(y[1:] - y[:-1])
    except (ValueError, IndexError):
        return 0


def maxGap(x):
    try:
        y = np.sort(x)
        return np.nanmax(y[1:] - y[:-1])
    except (ValueError, IndexError):
        return 0


def meanGap(x):
    try:
        y = np.sort(x)
        return np.nanmean(y[1:] - y[:-1])
    except (ValueError, IndexError):
        return 0


def CalculateNumericalStatsPandas(base, bNonZero=False):
    if bNonZero:
        r = base[base > 0].astype(float)
    else:
        r = base.astype(float)
    seriesMinimum = r.min()
    res = pd.Series(
        {
            "mean": r.mean(),
            "min": seriesMinimum,
            "max": r.max(),
            "std": r.std(ddof=0),
            "mad": r.mad(),
            "skew": r.skew(),
            "kurtosis": r.kurtosis(),
            "mode": r.mode().values[0],
            "percentile10": percentile10(r),
            "percentile50": percentile50(r),
            "percentile90": percentile90(r),
            "minGap": minGap(r),
            "maxGap": maxGap(r),
            "count_nonzero": np.count_nonzero(r),
            "numUnique": numUnique(r),
        }
    )
    if seriesMinimum > 0:
        res["geometric_mean"] = stats.geometric_mean(r)
        res["harmonic_mean"] = stats.harmonic_mean(r)
    else:
        res["geometric_mean"] = 0
        res["harmonic_mean"] = 0
    return res


def GetNumericalStatsNames():
    return "mean quadratic_mean geometric_mean harmonic_mean median mode min max mad std skew kurt per10 perc25 perc75 perc90 minGap maxGap meanGap count_nonzero numUnique slope weighted_mean first last ratio sum npositive ninteger".split(
        " "
    )
    """
    Добавить количество пересечений средних и медианного значений? (trend reversions)
    убрать гэпы. это статистика второго порядка и должна считаться отдельно. причем можно считать от разностей или от отношений.
    взвешенные статы считать отдельным вызовом ( и не только среднеарифметические, а ВСЕ).
    Добавить среднее кубическое, усечённое, 
    винзоризированное (https://ru.wikipedia.org/wiki/%D0%92%D0%B8%D0%BD%D0%B7%D0%BE%D1%80%D0%B8%D0%B7%D0%BE%D0%B2%D0%B0%D0%BD%D0%BD%D0%BE%D0%B5_%D1%81%D1%80%D0%B5%D0%B4%D0%BD%D0%B5%D0%B5).
    drawdowns, negative drawdowns (for shorts)
    """


def GetNumericalStatsNamesNanAware():
    return GetNumericalStatsNames() + [
        "num_nans",
    ]


def GetNumericalStatsNamesSmall():
    return ["mean", "median", "mode", "min", "max", "per10", "perc25", "perc75", "perc90", "count_nonzero", "numUnique"]


NUMERICAL_STATS_NAMES = GetNumericalStatsNames()
NUMERICAL_STATS_NAMES_NAN_AWARE = GetNumericalStatsNamesNanAware()


@numba.njit(fastmath=True)
def CalculateNumericalStatsSmall(x, l=0, r=0):
    if len(x) == 0:
        return 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    else:
        if r == 0:
            r = len(x) - 1
        fe = x[l]
        size = r + 1 - l

        # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # 1st pass
        # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

        i = 0
        lastV = 0.0
        mean = 0.0
        cnt_nonzero = 0
        maximum, minimum = fe, fe

        array_subset = x[l : r + 1]
        # print("slice=", array_subset)
        for next_value in array_subset:
            mean = mean + next_value
            if next_value > maximum:
                maximum = next_value
            elif next_value < minimum:
                minimum = next_value
            if not (next_value == 0.0):
                cnt_nonzero = cnt_nonzero + 1

            lastV = next_value
            i = i + 1
        mean = mean / size

        # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # 2nd pass
        # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

        # Sorting array

        xsorted = np.sort(array_subset)

        number = xsorted[0]
        mode = number
        numUnique = 1
        countMode = 1
        count = 1

        for t in xsorted:
            if t == number:
                count = count + 1
            else:
                numUnique = numUnique + 1
                if count > countMode:
                    countMode = count
                    mode = number
                number = t
                count = 1

        # compute median

        if size % 2 == 0:
            sent = int(size / 2)
            median = xsorted[sent - 1] + xsorted[sent]
        else:
            median = xsorted[int(size // 2)]
        factor = size / 100

        return (
            mean,
            xsorted[int(np.ceil(50 * factor)) - 1],
            mode,
            minimum,
            maximum,
            xsorted[int(np.ceil(10 * factor)) - 1],
            xsorted[int(np.ceil(25 * factor)) - 1],
            xsorted[int(np.ceil(75 * factor)) - 1],
            xsorted[int(np.ceil(90 * factor)) - 1],
            cnt_nonzero,
            numUnique,
        )


def CalculateNumericalStatsNumpyNumbaOptimizedNanAware(x, l=0, r=0, geomean_log_mode=0, weights=np.array([0.0])):
    if r == 0:
        r = len(x) - 1
    size = r + 1 - l
    mask = np.isnan(x[l:r])
    num_nans = np.sum(mask)
    if num_nans == 0:
        return CalculateNumericalStatsNumpyNumbaOptimized(x, l, r, geomean_log_mode, weights=weights) + tuple((num_nans,))
    elif num_nans == size:
        return tuple([np.nan] * len(GetNumericalStatsNames())) + tuple((num_nans,))
    else:
        if len(weights) < size:
            return CalculateNumericalStatsNumpyNumbaOptimized(x[l:r][~mask], geomean_log_mode=geomean_log_mode, weights=weights) + tuple((num_nans,))
        else:
            return CalculateNumericalStatsNumpyNumbaOptimized(x[l:r][~mask], geomean_log_mode=geomean_log_mode, weights=weights[~mask]) + tuple((num_nans,))


def CalculateCategoricalStatsNumpy(a, res, l=1):
    uniqueVals, freqs = np.unique(a, return_counts=True)
    idx = np.argsort(freqs)[::-1]
    for i in range(min(l, len(uniqueVals))):
        ind = idx[i]
        res[i] = uniqueVals[ind]
        res[i + l] = freqs[ind]

        # a = np.array([[1, 3, 4, 2, 2, 7],[5, 2, 2, 1, 4, 1],[3, 3, 2, 2, 1, 1]])
        # print(stats.mode(a,axis=None))
        # ModeResult(mode=array([2]), count=array([6]))

        # l=2
        # vals,counts=np.zeros(l,np.int32),np.zeros(l,np.int32)
        # ProcessCategoricals(a,vals,counts,l=l)
        # print(vals,counts)
        # [2 1] [6 5]

        # ************************************************************************************************************************************************************************************
        # Textual features
        # ************************************************************************************************************************************************************************************

        from random import randint, random


def flush_text_stats_caches():
    global charstats_buffer, oov_buffer, oov_tokens_buffer

    charstats_buffer = dict()
    oov_buffer = dict()
    oov_tokens_buffer = set()


def get_global_oov_tokens():
    return oov_tokens_buffer


def init_nlp(lib_name: str = "stanza", model_name: str = "en_core_web_sm", **kwargs):
    global nlp, nlp_lib, nlp_lang, nlp_feats_tag, nlp_fields, ner_fields, nlp_word_tags, nlp_ner_tags

    nlp_lib = lib_name
    nlp_lang = "english"  # default
    flush_text_stats_caches()
    ner_fields = ("text",)
    if nlp_lib == "spacy":
        #!pip install tf-nightly-gpu -U
        # pip install -U spacy[cuda110]
        # python -m spacy download en_core_web_sm
        # python -m spacy download ru_core_news_sm

        import spacy

        nlp = spacy.load(model_name)

        if model_name.startswith("en"):
            nlp_lang = "english"

        nlp_feats_tag = "morph"
        nlp_fields = ("text", "lemma_")
        nlp_word_tags = tuple("pos_ tag_ dep_ is_stop morph".split())  # is_alpha
        nlp_ner_tags = tuple("label_".split())

    elif nlp_lib == "stanza":
        #!pip install stanza
        import stanza

        kwargs["lang"] = kwargs.get("lang", "english")
        kwargs["processors"] = kwargs.get("processors", "tokenize,pos,ner,lemma,depparse")  # ,mwt,sentiment

        n = 0
        while True:
            try:
                nlp = stanza.Pipeline(**kwargs)  # use_gpu=True
            except Exception as e:
                n += 1
                if n > 2:
                    raise (e)
                stanza.download(kwargs["lang"])
            else:
                break

        nlp_lang = kwargs["lang"]

        nlp_feats_tag = "feats"
        nlp_fields = ("text", "lemma")
        nlp_word_tags = tuple("upos xpos deprel feats".split())
        nlp_ner_tags = tuple("type".split())

    load_stopwords(lang=nlp_lang)
    if domain_suffixes is None:
        get_domain_suffixes()


def get_domain_suffixes() -> tuple:
    global domain_suffixes
    import requests

    res = None
    try:
        res = requests.get("https://publicsuffix.org/list/public_suffix_list.dat")
        logging.info(f"Downloaded internet domain suffixes list")
    except Exception as e:
        logging.exception(e)
    if res is None:
        try:
            with open("public_suffix_list.dat") as f:
                res = f.read()
            logging.info(f"Loaded internet domain suffixes list")
        except Exception as e:
            logging.warning(f"Could not loaded internet domain suffixes list from both web and local file")
    if res:
        lst = set()
        for line in res.text.split("\n"):
            if not line.startswith("//"):
                domains = line.split(".")
                cand = domains[-1]
                if cand:
                    lst.add("." + cand)

        domain_suffixes = tuple(sorted(lst))

        return domain_suffixes


def load_stopwords(lang: str = "english") -> set:
    global nlp_stopwords

    #!pip install advertools
    import advertools as adv

    try:
        nlp_stopwords = set(sorted(adv.stopwords[lang]))
        logging.info(f"Loaded stopwords for {lang}")
        return True
    except Exception as e:
        logging.warning(f"Could not load stopwords for {lang}")

    return nlp_stopwords


def is_stopword(txt: str) -> bool:
    return txt in nlp_stopwords


def is_hashtag(txt: str) -> bool:
    return txt.startswith("#")


def is_mention(txt: str) -> bool:
    return txt.startswith("@")


def reminds_url(txt: str) -> bool:
    """
    >>> reminds_url('yandex.ru.com/somepath')
    True

    """
    ltext = txt.lower().split("/")[0]
    return ltext.startswith(("http", "www", "ftp")) or ltext.endswith(domain_suffixes)


nlp_funcs = [getattr(str, func) for func in "isalnum isalpha isdecimal isdigit isidentifier islower isnumeric isprintable istitle isupper".split()]
nlp_funcs += [is_hashtag, is_mention, reminds_url, is_stopword]


def get_words_stream(nlp_obj: object) -> str:
    if nlp_lib == "spacy":
        for token in nlp_obj:
            yield token
    elif nlp_lib == "stanza":
        for sentence in nlp_obj.sentences:
            for token in sentence.words:
                yield token


def get_sentences(nlp_obj: object):
    if nlp_lib == "spacy":
        return nlp_obj.sents
    elif nlp_lib == "stanza":
        return nlp_obj.sentences


def get_words(nlp_obj: object):
    if nlp_lib == "spacy":
        return nlp_obj
    elif nlp_lib == "stanza":
        return nlp_obj.words


def get_entities(nlp_obj: object):
    if nlp_lib == "spacy":
        return nlp_obj.ents
    elif nlp_lib == "stanza":
        return nlp_obj.ents


def get_nlp_tag(word: object, tag_name: str) -> str:
    if tag_name == nlp_feats_tag:
        tags = getattr(word, tag_name)
        if tags is not None:
            for tag in tags.split("|"):
                yield tag
    else:
        yield getattr(word, tag_name)


def get_count_stats(counts: dict, cnt: int, cntunique: int, keys: Sequence = None, main_key: str = None, prefix: str = "") -> dict:
    """
    Returns simple count/unique count/pct stats for grouped stats
    """
    if counts is None:
        return
    res = {}
    if keys is None:
        keys = counts.keys()

    if main_key is not None:
        if main_key in keys:
            prefix = main_key + "_"
            cnt = len(counts[main_key])
            cntunique = len(set(counts[main_key]))

    if cnt > 0 and cntunique > 0:
        res = dict()

        res[prefix + "cnt"] = cnt
        res[prefix + "cntunique"] = cntunique
        res[prefix + "pctunique"] = cntunique / cnt
        for key in keys:
            if main_key is not None:
                if key == main_key:
                    continue  # skipping main key

            res[key + "_cnt"] = len(counts[key])
            res[key + "_cntunique"] = len(set(counts[key]))

            res[key + "_pctunique"] = res[key + "_cntunique"] / res[key + "_cnt"]

            if key != "wd_lemma":
                res[key + "_pctoverall"] = res[key + "_cnt"] / cnt
            res[key + "_pctoverallunique"] = res[key + "_cntunique"] / cntunique
    return res


def get_sum_stats(sums: dict, sm: int, keys: Sequence = None, main_key: str = None, prefix: str = "") -> dict:
    """
    Returns simple sum/pct stats for grouped stats
    """
    if sums is None:
        return
    res = {}
    if keys is None:
        keys = sums.keys()

    if main_key is not None:
        if main_key in keys:
            prefix = main_key + "_"
            sm = sums[main_key]

    if sm > 0:
        res = dict()

        res[prefix + "sum"] = sm
        for key in keys:
            if main_key is not None:
                if key == main_key:
                    continue  # skipping main key

            res[key + "_sum"] = sums[key]

            if key != "wd_lemma":
                res[key + "_pctsumoverall"] = res[key + "_sum"] / sm
    return res


def get_char_stats(sent: str, max_buf_len: int = 30) -> dict:
    """
    Returns counts, unique counts, percentages of various character groups in text:
    by case, punctuation or not, vowel or not.
    """
    global charstats_buffer

    if len(sent) <= max_buf_len and (sent in charstats_buffer):
        res = charstats_buffer[sent]
    else:
        counts = defaultdict(list)
        for char in sent:
            if char in punctuation:  #'!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
                counts["punctuation"].append(char)
            else:
                if char.isnumeric():
                    counts["numeric"].append(char)
                else:
                    if char in VOWELS:
                        counts["vowels"].append(char)
                    else:
                        counts["consonants"].append(char)

                    if char.isupper():
                        counts["upper"].append(char)
                    else:
                        counts["lower"].append(char)
        res = get_count_stats(counts=counts, cnt=len(sent), cntunique=len(set(sent)))
        if len(sent) <= max_buf_len:
            charstats_buffer[sent] = res

    return res


def in_vocabulary(texts: list, speller: object) -> tuple:
    if len(speller.existing(texts)) > 0:
        return 1, None
    else:
        suggestions = speller.get_candidates(texts[0])
        for suggestion in suggestions:
            if suggestion[0] == 0:
                return 0, None
            else:
                return 0, suggestion[1]


def add_stats(key: str, token: str, ltoken: str, attrib: str, char_stats: dict, sentword_counts: dict, sentword_sums: dict) -> None:
    if attrib == "text":
        sentword_counts[key].append(token)
        if ltoken is not None:
            sentword_counts[key + "_lw"].append(ltoken)
    else:
        sentword_counts[key].append(ltoken)

    # nlp_funcs & char stats: apply JUST to text, NOT lemmas
    if attrib == "text":
        if char_stats is not None:
            for charprop, val in get_char_stats(token).items():
                char_stats[key + "_chars_" + charprop].append(val)
        if sentword_sums is not None:
            for func in nlp_funcs:
                sentword_sums[key + "_" + func.__name__] += func(token)  # 1 or 0


def create_textual_features(
    text: str,
    doc: object = None,
    word_tags: Sequence = None,
    ner_tags: Sequence = None,
    return_ner_stats: bool = True,
    return_char_stats: str = "",
    return_sent_stats: bool = True,
    return_lowercase_stats: bool = False,
    return_wordsum_stats: bool = True,
    return_entropy_order: int = None,
    return_oov_stats: str = "",
    return_unique_tags: bool = True,
    return_oov_tokens: bool = False,
    count_sentences: bool = False,
) -> dict:
    global charstats_buffer, oov_buffer, oov_tokens_buffer
    global spellers, langs

    # Checks
    assert return_char_stats in (None, "", "text", "sentences", "words", "all")

    if return_oov_stats:
        oov_tokens = set()
        requested_langs = set(return_oov_stats.split())
        assert requested_langs.issubset("en, pl, ru, uk, tr, es, pt, cs".split(", "))

    # Inits
    ltoken = None
    nsentences = 0
    nwords = 0
    nvalidwords = 0

    if return_oov_stats:
        iv_sums = defaultdict(int)
        iv_sums["wd_oov"] = 0
        iv_sums["spellcorrected"] = 0
        if requested_langs != langs:
            #!pip install autocorrect
            logging.info("creating spellers...")
            from autocorrect import Speller

            spellers = {}
            langs = requested_langs
            for lang in langs:
                spellers[lang] = Speller(lang)
            logging.info("created")

    if return_char_stats:
        char_stats = defaultdict(list)
    else:
        char_stats = None
    wordcount_stats = defaultdict(list)
    wordsum_stats = defaultdict(list)
    word_counts = defaultdict(list)
    word_sums = defaultdict(int)

    if word_tags is None:
        word_tags = nlp_word_tags
    if ner_tags is None:
        ner_tags = nlp_ner_tags

    # Parsing by a NLP engine
    offset = 0
    if doc is None:  # just text.
        while True:
            try:
                doc = nlp(text)
            except Exception as e:
                logging.error(f"Error {e} at text {text}")
                if "stack expects a non-empty TensorList" in str(e):
                    if len(text) > 5:
                        text = text[:-1]
                    else:
                        raise (e)
                else:
                    raise (e)
            else:
                break
        all_sentences = get_sentences(doc)
    else:
        all_sentences = [doc]

    # for word in get_words_stream(doc):

    for sentence in all_sentences:
        if return_sent_stats:
            # this can be made more efficient later
            if return_char_stats in ("sentences", "all"):
                for charprop, val in get_char_stats(sentence.text).items():
                    char_stats["sent_chars_" + charprop].append(val)

        sentword_counts = defaultdict(list)
        if return_wordsum_stats:
            sentword_sums = defaultdict(int)
        else:
            sentword_sums = None
        nsentwords = 0

        # Words
        universe = [("wd", get_words, nlp_fields, word_tags)]
        if return_ner_stats:
            # NamedEntities
            universe.append(("ne", get_entities, ner_fields, ner_tags))

        for level, get_func, allowed_fields, tags in universe:
            for word in get_func(sentence):
                # By text and lemma
                for attrib in allowed_fields:
                    token = getattr(word, attrib).strip()
                    if len(token) > 1 and token.endswith("."):
                        token = token[:-1]

                    if not (token in punctuation):
                        if level == "wd":
                            if attrib == "text":
                                nsentwords += 1

                        if return_lowercase_stats:
                            ltoken = token.lower()
                        else:
                            if attrib == "text":
                                ltoken = None
                            else:
                                ltoken = token.lower()

                        if return_oov_stats:
                            if level == "wd":
                                if attrib == "text":
                                    if not (token in punctuation or token.isnumeric()):
                                        nvalidwords += 1
                                        if token in oov_buffer:
                                            is_oov, any_spellcorrected = oov_buffer[token]
                                        else:
                                            texts = [token]
                                            is_oov = True
                                            any_spellcorrected = False
                                            for lang in langs:
                                                token_in_vocabulary, suggestion = in_vocabulary(texts=texts, speller=spellers[lang])  # 1 or 0
                                                # iv_sums['wd_iv_'+lang]+=token_in_vocabulary
                                                if token_in_vocabulary == 1:
                                                    is_oov = False
                                                    break
                                                else:
                                                    if suggestion is not None:
                                                        any_spellcorrected = True
                                            oov_buffer[token] = (is_oov, any_spellcorrected)
                                        if is_oov:
                                            iv_sums["wd_oov"] += 1
                                            if any_spellcorrected:
                                                iv_sums["spellcorrected"] += 1
                                            oov_tokens.add(token)

                        # regardless of POS
                        key = "_".join([level, attrib])
                        add_stats(key, token, ltoken, attrib, char_stats if return_char_stats in ("words", "all") else None, sentword_counts, sentword_sums)
                        # By POS
                        if attrib != "text":
                            for tag_name in tags:
                                for tag in get_nlp_tag(word, tag_name):
                                    if tag is not None:
                                        # if 'lemma' in attrib:
                                        #    if tag not in ('ADJ','ADV','NOUN','PRON','VERB'):
                                        #        continue
                                        key = "_".join([level, attrib, tag])
                                        add_stats(
                                            key,
                                            token,
                                            ltoken,
                                            attrib,
                                            char_stats if return_char_stats in ("words", "all") else None,
                                            sentword_counts,
                                            sentword_sums,
                                        )

        # Word stats (count & sum) aggregated for this sentence
        if return_sent_stats:
            for wordprop, val in get_count_stats(counts=sentword_counts, cnt=0, cntunique=0, main_key="wd_text").items():
                wordcount_stats["sent_" + wordprop].append(val)

            if return_wordsum_stats:
                for wordprop, val in get_sum_stats(sums=sentword_sums, sm=nsentwords).items():
                    wordsum_stats["sent_" + wordprop] = val

        # append sent stats to overall text stats
        for key, val in sentword_counts.items():
            word_counts[key].extend(val)

        if return_wordsum_stats:
            for key, val in sentword_sums.items():
                word_sums[key] += sentword_sums[key]

        nsentences += 1
        nwords += nsentwords

    stats = dict()
    # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # Char stats finalization
    # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    if return_char_stats in ("text", "all"):
        # for whole text (simple stats)
        for charprop, val in get_char_stats(text).items():
            stats["chars_" + charprop] = val

    if return_char_stats in ("sentences", "words", "all"):
        # aggregated over sentences & words
        for key, stats_list in char_stats.items():
            for b, a in zip(CalculateNumericalStatsNumpyNumbaOptimized(np.array(stats_list)), NUMERICAL_STATS_NAMES):
                stats[key + "_" + a] = b

    # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # Word stats finalization
    # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    # for whole text (simple stats)

    for wordprop, val in get_count_stats(counts=word_counts, cnt=0, cntunique=0, main_key="wd_text").items():
        stats[wordprop] = val

    if return_wordsum_stats:
        for wordprop, val in get_sum_stats(sums=word_sums, sm=nwords, prefix="wd_text_").items():
            stats[wordprop] = val

    if return_oov_stats:
        for wordprop, val in get_sum_stats(sums=iv_sums, sm=nvalidwords, prefix="valid_wd_").items():
            stats[wordprop] = val

    if return_sent_stats:
        # aggregated over sentences
        for key, stats_list in wordcount_stats.items():
            for b, a in zip(CalculateNumericalStatsNumpyNumbaOptimized(np.array(stats_list)), NUMERICAL_STATS_NAMES):
                stats[key + "_" + a] = b

    if count_sentences:
        stats["nsentences"] = nsentences
    # stats['nwords']=nwords

    if return_entropy_order is not None:
        if return_entropy_order >= 0:
            for order in range(return_entropy_order + 1):
                entropy, entropy_rate = compute_entropy_stats(text, order)
                stats["entropy_" + str(order)] = entropy
                stats["entropy_rate_" + str(order)] = entropy_rate

    res = [stats]

    if return_unique_tags:
        tail = "_lw" if return_lowercase_stats else ""
        unique_words, unique_lemmas = [set(word_counts["wd_" + field + tail]) for field in nlp_fields]
        res.append(unique_words)
        res.append(unique_lemmas)

        if return_ner_stats:
            unique_ners = [set(word_counts["ne_" + field + tail]) for field in ner_fields][0]
            res.append(unique_ners)

    if return_oov_tokens:
        res.append(oov_tokens)

    if return_oov_stats:
        if len(oov_tokens) > 0:
            oov_tokens_buffer.update(oov_tokens)

    return res


columns_order = [
    "char_capitals",
    "char_numeric",
    "char_total",
    "char_vowels",
    "dep_ROOT",
    "dep_acl",
    "dep_acomp",
    "dep_advcl",
    "dep_advmod",
    "dep_agent",
    "dep_amod",
    "dep_appos",
    "dep_attr",
    "dep_aux",
    "dep_auxpass",
    "dep_case",
    "dep_cc",
    "dep_ccomp",
    "dep_compound",
    "dep_conj",
    "dep_csubj",
    "dep_dative",
    "dep_dep",
    "dep_det",
    "dep_dobj",
    "dep_expl",
    "dep_intj",
    "dep_mark",
    "dep_meta",
    "dep_neg",
    "dep_nmod",
    "dep_npadvmod",
    "dep_nsubj",
    "dep_nsubjpass",
    "dep_nummod",
    "dep_oprd",
    "dep_parataxis",
    "dep_pcomp",
    "dep_pobj",
    "dep_poss",
    "dep_preconj",
    "dep_predet",
    "dep_prep",
    "dep_prt",
    "dep_punct",
    "dep_quantmod",
    "dep_relcl",
    "dep_xcomp",
    "is_hashtag",
    "is_mention",
    "is_stop_False",
    "is_stop_True",
    "isalnum",
    "isalpha",
    "isdecimal",
    "isdigit",
    "isidentifier",
    "islower",
    "isnumeric",
    "isprintable",
    "istitle",
    "isupper",
    "ner_CARDINAL",
    "ner_DATE",
    "ner_EVENT",
    "ner_FAC",
    "ner_GPE",
    "ner_LANGUAGE",
    "ner_LAW",
    "ner_LOC",
    "ner_MONEY",
    "ner_NORP",
    "ner_ORDINAL",
    "ner_ORG",
    "ner_PERCENT",
    "ner_PERSON",
    "ner_PRODUCT",
    "ner_QUANTITY",
    "ner_TIME",
    "ner_WORK_OF_ART",
    "nerwdlen_count_nonzero",
    "nerwdlen_gmean",
    "nerwdlen_hmean",
    "nerwdlen_kurt",
    "nerwdlen_mad",
    "nerwdlen_max",
    "nerwdlen_maxGap",
    "nerwdlen_mean",
    "nerwdlen_meanGap",
    "nerwdlen_median",
    "nerwdlen_min",
    "nerwdlen_minGap",
    "nerwdlen_mode",
    "nerwdlen_numUnique",
    "nerwdlen_per10",
    "nerwdlen_perc25",
    "nerwdlen_perc75",
    "nerwdlen_perc90",
    "nerwdlen_qmean",
    "nerwdlen_ratio",
    "nerwdlen_skew",
    "nerwdlen_slope",
    "nerwdlen_std",
    "nerwdlen_weighted_mean",
    "pos_ADJ",
    "pos_ADP",
    "pos_ADV",
    "pos_AUX",
    "pos_CCONJ",
    "pos_DET",
    "pos_INTJ",
    "pos_NOUN",
    "pos_NUM",
    "pos_PART",
    "pos_PRON",
    "pos_PROPN",
    "pos_PUNCT",
    "pos_SCONJ",
    "pos_SYM",
    "pos_VERB",
    "pos_X",
    "tag_$",
    "tag_''",
    "tag_,",
    "tag_-LRB-",
    "tag_-RRB-",
    "tag_.",
    "tag_:",
    "tag_ADD",
    "tag_CC",
    "tag_CD",
    "tag_DT",
    "tag_EX",
    "tag_FW",
    "tag_HYPH",
    "tag_IN",
    "tag_JJ",
    "tag_JJR",
    "tag_JJS",
    "tag_LS",
    "tag_MD",
    "tag_NFP",
    "tag_NN",
    "tag_NNP",
    "tag_NNPS",
    "tag_NNS",
    "tag_PDT",
    "tag_POS",
    "tag_PRP",
    "tag_PRP$",
    "tag_RB",
    "tag_RBR",
    "tag_RBS",
    "tag_RP",
    "tag_SYM",
    "tag_TO",
    "tag_UH",
    "tag_VB",
    "tag_VBD",
    "tag_VBG",
    "tag_VBN",
    "tag_VBP",
    "tag_VBZ",
    "tag_WDT",
    "tag_WP",
    "tag_WRB",
    "tag_XX",
    "tag_``",
    "wdlen_count_nonzero",
    "wdlen_gmean",
    "wdlen_hmean",
    "wdlen_kurt",
    "wdlen_mad",
    "wdlen_max",
    "wdlen_maxGap",
    "wdlen_mean",
    "wdlen_meanGap",
    "wdlen_median",
    "wdlen_min",
    "wdlen_minGap",
    "wdlen_mode",
    "wdlen_numUnique",
    "wdlen_per10",
    "wdlen_perc25",
    "wdlen_perc75",
    "wdlen_perc90",
    "wdlen_qmean",
    "wdlen_ratio",
    "wdlen_skew",
    "wdlen_slope",
    "wdlen_std",
    "wdlen_weighted_mean",
]


def featurize_text_list(lst: list, to_numpy: bool = True, sort_columns: bool = True) -> object:
    tdf = pd.DataFrame(map(create_textual_features, lst)).fillna(0)
    # insert zeroes instead of missing columns
    if sort_columns:
        columns = tdf.columns
        for col in columns_order:
            if col not in columns:
                tdf[col] = 0.0

        tdf = tdf[columns_order]
    if to_numpy:
        return tdf.values
    else:
        return tdf


def featurize_text_df(df, col: int = 0, to_numpy: bool = True) -> object:
    return featurize_text_list(df.iloc[:, col].to_list(), to_numpy=to_numpy)


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Text altering operators
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


def drop_vowels_and_merge_words(text: str, unvoweling_prob: float = 0.8, max_group_size: int = 6) -> str:
    import nltk

    # clamp random subgroups of words together
    words = nltk.word_tokenize(text)
    n = 0
    l = len(words)
    res = []
    while n < l:
        k = min(randint(1, max_group_size), l - n)
        subgroup = []
        for i in range(k):
            if random() < unvoweling_prob:
                candidate = "".join([l for l in words[n + i] if l not in VOWELS])
            else:
                candidate = words[n + i]
            subgroup.append(candidate)
        n += k
        res.append("".join(subgroup))
    return " ".join(res)


# ************************************************************************************************************************************************************************************
# Pipeline members
# ************************************************************************************************************************************************************************************

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Специализированные трансформаторы
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

from sklearn.base import BaseEstimator


# @numba.jit()
def calcStats(vals, y, target_stats):
    unique_vals, counts = np.unique(vals, return_counts=True)
    maxVal = int(np.max(unique_vals)) + 1
    catStats = np.empty((maxVal, len(target_stats)), float)
    catCounts = np.zeros(maxVal, float)

    # Traverse each category of that feature
    for i, (val, cnt) in enumerate(zip(unique_vals, counts)):
        target = y[vals == val]
        catStats[val, :] = [statFunc(target) for statFunc in target_stats]
        catCounts[val] = cnt / len(vals)
    return [catStats, catCounts]


class TargetEncodingTransformer(BaseEstimator):
    def __init__(
        self,
        cat_columns="all",
        num_columns=[],
        interactions_degree=1,
        target_stats=[np.mean],
        cat_low_thresh=0.1,
        num_nbins=10,
        noise=0,
        smooth=False,
        verbose=1,
    ):
        import inspect

        # np.min,np.max,percentile50
        args, _, _, values = inspect.getargvalues(inspect.currentframe())
        values.pop("self")
        # store parameters as public attributes
        for arg, val in values.items():
            setattr(self, arg, val)

    def fit(self, X, y=None, **fit_params):
        # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # Postponed (as per sklearn's rules) validation of init parameters
        # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if False:
            allowed_stats = [np.mean]
            for next_stat in self.target_stats:
                if next_stat not in allowed_stats:
                    raise ValueError("statistic must be in " + str(allowed_stats) + ", " + next_stat + " is wrong")
        # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # Getting fingerprint of the train dataset
        # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        self.ncolumns_ = X.shape[1]

        # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # Categorical features
        # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

        # Global stats for target variable
        global_stats = np.empty(len(self.target_stats), float)
        for i, statFunc in enumerate(self.target_stats):
            global_stats[i] = statFunc(y)

        # Now let's consider each categorical level of every variable individually
        stats = []
        if self.cat_columns == "all":
            cat_columns = np.arange(X.shape[1])
        else:
            cat_columns = self.cat_columns
        for j, col in enumerate(cat_columns):
            stats.append(calcStats(X[:, col], y, self.target_stats))
            if self.verbose > 1:
                print("Finished categorical fitting on variable %d/%d" % (j + 1, len(cat_columns)))
        self.global_stats_ = global_stats
        self.stats_ = stats
        if self.verbose > 0:
            print("Finished categorical fitting of %d variable(s)" % len(cat_columns))

        return self

    # @numba.jit()
    def transform(self, X, **transform_params):
        import numpy as np

        # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # check that we have an input matrix with same fingerprint (size, columns names) as the one we fit on
        # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if self.ncolumns_ != X.shape[1]:
            raise ValueError("Passed DataFrame has different number of columns (%s) than the DataFrame we fit on (%s)!" % (self.ncolumns_, X.shape[1]))

        nCols = len(self.target_stats)
        if self.cat_columns == "all":
            cat_columns = np.arange(X.shape[1])
        else:
            cat_columns = self.cat_columns
        data = np.empty((len(X), len(cat_columns) * nCols), float)
        # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # Traverse every categorical variable
        # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        global_stats = np.array(self.global_stats_)
        for j, col in enumerate(cat_columns):
            unique_vals, counts = np.unique(X[:, col], return_counts=True)
            [catStats, catCounts] = self.stats_[j]
            maxVal = len(catCounts) - 1
            sl = slice(j * nCols, (j + 1) * nCols)
            for i, (val, count) in enumerate(zip(unique_vals, counts)):
                if val <= maxVal and catCounts[val] > 0:
                    if self.smooth:
                        if self.noise > 0:
                            w = catCounts[val] * (1 + np.random.randn(count, nCols) * self.noise)
                        else:
                            w = catCounts[val]
                        data[X[:, col] == val, sl] = catStats[val] * w + global_stats * (1 - w)
                    else:
                        if self.noise > 0:
                            data[X[:, col] == val, sl] = catStats[val] * (1 + np.random.randn(count, nCols) * self.noise)
                        else:
                            data[X[:, col] == val, sl] = catStats[val]
                else:
                    if self.verbose > 1:
                        print("While transforming column %d, for group %d population mean target of %.3f was used" % (col, val, global_stats))
                    if self.noise > 0:
                        data[X[:, col] == val, sl] = global_stats * (1 + np.random.randn(count, nCols) * self.noise)
                    else:
                        data[X[:, col] == val, sl] = global_stats
            if self.verbose > 1:
                print("Finished applying categorical transform to variable %d/%d" % (j + 1, len(cat_columns)))
        # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        # Now let's append columns which don't need touching
        # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        if X.shape[1] > len(cat_columns):
            data = np.concatenate([data, X[:, list(set(np.arange(X.shape[1])) - set(cat_columns))]], axis=0)
        if self.verbose > 1:
            print("Finished categorical transform of %d variable(s)" % len(cat_columns))
        return data

    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, y, **fit_params).transform(X)

    def get_feature_names():
        return self._feature_names


flush_text_stats_caches()
