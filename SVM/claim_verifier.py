# basic deps ------------------------------------------------
import pandas as pd, numpy as np, re, string, time, joblib, nltk  # core libs
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# sklearn ------------------------------------------------------
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC, LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.kernel_approximation import RBFSampler, Nystroem

nltk.download("stopwords", quiet=True)

# -------------------------------------------------------------
class ClaimVerifier:
    """tiny liar‑dataset fact‑checker"""

    def __init__(self):
        neg = {"no", "not", "never", "none", "nobody", "nothing", "nowhere", "neither", "nor"}
        sw  = set(stopwords.words("english"))
        self.stop = sw - neg  # keep negation words
        self.stem = PorterStemmer()  # porter stem
        self.tfidf = self.pipe = self.svm = None

    # --- text helpers -----------------------------------------
    def _clean(self, txt: str) -> str:
        txt = txt.lower()
        txt = re.sub(r"http\S+", "", txt)            # drop urls
        # strip punctuation except ?!
        txt = re.sub(r"[%s]" % re.escape(string.punctuation.replace("?", "").replace("!", "")), " ", txt)
        txt = txt.replace("?", " qmark ").replace("!", " bang ")
        words = [w for w in txt.split() if w not in self.stop]
        words = [self.stem.stem(w) for w in words]
        return " ".join(words)

    @staticmethod
    def _bin(lbl: str) -> int:  # six‑way → binary
        return 1 if lbl in {"true", "mostly-true"} else 0

    def _extra(self, df: pd.DataFrame):
        """basic numeric fe's"""
        df["word_cnt"] = df["statement"].str.split().str.len()
        df["char_cnt"] = df["statement"].str.len()
        df["q_cnt"]   = df["statement"].str.count(r"\?")
        df["bang_cnt"] = df["statement"].str.count(r"!")
        df["num_cnt"]  = df["statement"].str.count(r"\d")
        # ratio of caps
        df["cap_ratio"] = df["statement"].apply(lambda x: sum(c.isupper() for c in x)/len(x) if x else 0)
        return df

    # --- io ----------------------------------------------------
    def load_liar(self, train_p: str, test_p: str, valid_p: str | None = None):
        cols = [
            "id", "label", "statement", "subject", "speaker", "job", "state", "party",
            "barely_true", "false", "half_true", "mostly_true", "pants_fire", "context",
        ]
        tr = pd.read_csv(train_p, sep="\t", header=None, names=cols)
        te = pd.read_csv(test_p,  sep="\t", header=None, names=cols)
        va = pd.read_csv(valid_p, sep="\t", header=None, names=cols) if valid_p else None

        for df in (tr, te, *( [va] if va is not None else [] )):
            df["txt"] = df["statement"].apply(self._clean)
            df["lbl"] = df["label"].apply(self._bin)
            self._extra(df)
        return tr, te, va

    # --- model build ------------------------------------------
    def _build(self):
        self.tfidf = TfidfVectorizer(max_features=5000, min_df=5, max_df=0.7,
                                     ngram_range=(1,2), sublinear_tf=True)
        self.pipe = Pipeline([
            ("tfidf", self.tfidf),
            ("kbest", SelectKBest(chi2, k=1000)),
            ("svd", TruncatedSVD(n_components=100)),
            ("scale", StandardScaler()),
            ("svm", SVC(probability=True)),  # main classifier
        ])

    # --- training ---------------------------------------------
    def train(self, df: pd.DataFrame):
        if self.pipe is None:
            self._build()
        x, y = df["txt"], df["lbl"]
        grid = {  # slim search
            "svm__C": [1, 10],
            "svm__kernel": ["linear", "rbf"],
            "svm__gamma": ["scale", 0.1],
            "kbest__k": [1000],
        }
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        gs = GridSearchCV(self.pipe, grid, cv=cv, scoring="f1", n_jobs=-1)
        gs.fit(x, y)
        self.pipe = gs.best_estimator_;  self.svm = self.pipe.named_steps["svm"]
        if self.svm.kernel == "rbf":
            self._approx(x, y, cv)  # maybe speedup

    # --- rbf approx -------------------------------------------
    def _approx(self, x, y, cv):
        base = np.mean(cross_val_score(self.pipe, x, y, cv=cv, scoring="f1"))
        cfgs = {
            "rbf_sampler": RBFSampler(gamma=self.svm.gamma, n_components=100, random_state=42),
            "nystroem": Nystroem(gamma=self.svm.gamma, n_components=100, random_state=42),
        }
        best = None
        for tag, approx in cfgs.items():
            tmp = Pipeline([
                ("tfidf", self.tfidf),
                ("kbest", self.pipe.named_steps["kbest"]),
                ("svd",   self.pipe.named_steps["svd"]),
                ("scale", self.pipe.named_steps["scale"]),
                (tag, approx),
                ("lin", LinearSVC(C=self.svm.C)),
            ])
            sc = np.mean(cross_val_score(tmp, x, y, cv=cv, scoring="f1"))
            if sc > base * 0.98:  # keep if near‑equal
                best, base = tmp, sc
        if best:
            self.pipe = best

    # --- evaluation -------------------------------------------
    def eval(self, df: pd.DataFrame):
        x, y = df["txt"], df["lbl"]
        preds = self.pipe.predict(x)
        print("acc", accuracy_score(y, preds).round(3))  # quick acc
        print(classification_report(y, preds, digits=3))
        print(confusion_matrix(y, preds))
        return preds

    # --- single claim ----------------------------------------
    def verify(self, claim: str):  # small api
        clean = self._clean(claim)
        if hasattr(self.svm, "predict_proba"):
            prob = self.pipe.predict_proba([clean])[0]
            lbl, conf = ("true", prob[1]) if prob[1] >= 0.5 else ("false", prob[0])
        else:  # linear svc path
            score = float(self.pipe.decision_function([clean])[0])
            lbl, conf = ("true", abs(score)) if score >= 0 else ("false", abs(score))
        return lbl, conf

    # --- persistence -----------------------------------------
    def save(self, path: str):
        joblib.dump(self.pipe, path)

    def load(self, path: str):
        self.pipe = joblib.load(path)
        self.svm  = self.pipe.named_steps.get("svm")

# -------------------------------------------------------------
if __name__ == "__main__":
    cv = ClaimVerifier()
    tr, te, _ = cv.load_liar("Liar/train.tsv", "Liar/test.tsv")
    cv.train(tr)
    cv.eval(te)

    # smoke test
    for sentence in [
        "Building a wall on the U.S.-Mexico border will take literally years.",
        "The number of illegal immigrants could be 3 million. It could be 30 million.",
    ]:
        print(sentence, "->", cv.verify(sentence))
