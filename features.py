"""
features.py
------------
Lexical and host-based feature extraction for phishing URL detection.

The feature set is based on well-established indicators used in phishing
detection literature (e.g., the UCI "Phishing Websites" dataset feature
schema and subsequent academic work), covering:

  1. Lexical features   - derived purely from the URL string (always available,
                           work fully offline).
  2. Host-based features - DNS resolution / certificate presence (require a live
                           network connection; degrade gracefully to a neutral
                           value when unavailable, e.g. in offline/sandboxed
                           environments).

Every feature is normalised to be model-friendly (numeric, mostly 0/1 flags
or small integers) and returned alongside a human-readable explanation so the
web UI can show *why* a URL was flagged, not just the verdict.
"""

import re
import socket
import ssl
from urllib.parse import urlparse

SUSPICIOUS_WORDS = [
    "login", "verify", "secure", "account", "update", "confirm", "bank",
    "signin", "password", "billing", "webscr", "ebayisapi", "suspend",
    "limited", "alert", "recover", "unlock", "wallet",
]

SHORTENING_SERVICES = [
    "bit.ly", "goo.gl", "tinyurl.com", "t.co", "ow.ly", "is.gd", "buff.ly",
    "adf.ly", "shorte.st", "cutt.ly", "rebrand.ly", "tiny.cc",
]

SUSPICIOUS_TLDS = [
    "tk", "ml", "ga", "cf", "gq", "xyz", "top", "work", "click", "loan",
    "download", "gdn", "men", "review",
]

IP_PATTERN = re.compile(
    r"^(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})$"
)


def _has_ip_host(hostname):
    if not hostname:
        return False
    return bool(IP_PATTERN.match(hostname))


def _count(pattern, s):
    return len(re.findall(pattern, s))


def extract_lexical_features(url):
    """Extract features derivable purely from the URL string (no network)."""
    url = url.strip()
    parsed = urlparse(url if "://" in url else "http://" + url)
    hostname = parsed.hostname or ""
    path = parsed.path or ""
    full = url.lower()

    subdomain_count = max(hostname.count(".") - 1, 0) if hostname else 0
    tld = hostname.split(".")[-1].lower() if "." in hostname else ""

    feats = {
        "url_length": len(url),
        "hostname_length": len(hostname),
        "path_length": len(path),
        "num_dots": hostname.count("."),
        "num_hyphens": _count(r"-", url),
        "num_underscores": _count(r"_", url),
        "num_slashes": _count(r"/", path),
        "num_digits": _count(r"\d", url),
        "num_special_chars": _count(r"[@%=&\?\$!#\*]", url),
        "num_subdomains": subdomain_count,
        "digit_ratio": round(_count(r"\d", url) / max(len(url), 1), 3),
        "has_ip_host": int(_has_ip_host(hostname)),
        "has_at_symbol": int("@" in url),
        "has_double_slash_redirect": int("//" in path),
        "has_https_token_in_domain": int("https" in hostname.replace("www.", "")),
        "is_shortened": int(any(s in hostname for s in SHORTENING_SERVICES)),
        "has_suspicious_word": int(any(w in full for w in SUSPICIOUS_WORDS)),
        "suspicious_word_count": sum(1 for w in SUSPICIOUS_WORDS if w in full),
        "has_suspicious_tld": int(tld in SUSPICIOUS_TLDS),
        "has_port": int(parsed.port is not None),
        "uses_https": int(parsed.scheme == "https"),
        "hostname_has_digits": int(bool(re.search(r"\d", hostname))),
    }
    return feats, hostname, parsed.scheme


def extract_host_features(hostname, timeout=1.5):
    """
    Host-based features that require a live network connection.
    Fails gracefully (neutral default values) when offline or the host
    cannot be reached -- this keeps the tool usable in restricted/offline
    environments while still adding signal when connectivity is available.
    """
    feats = {"dns_resolves": 0, "has_valid_tls": 0}
    if not hostname:
        return feats
    try:
        socket.setdefaulttimeout(timeout)
        socket.gethostbyname(hostname)
        feats["dns_resolves"] = 1
    except Exception:
        feats["dns_resolves"] = 0

    try:
        ctx = ssl.create_default_context()
        with socket.create_connection((hostname, 443), timeout=timeout) as sock:
            with ctx.wrap_socket(sock, server_hostname=hostname) as ssock:
                ssock.getpeercert()
                feats["has_valid_tls"] = 1
    except Exception:
        feats["has_valid_tls"] = 0

    return feats


FEATURE_ORDER = [
    "url_length", "hostname_length", "path_length", "num_dots", "num_hyphens",
    "num_underscores", "num_slashes", "num_digits", "num_special_chars",
    "num_subdomains", "digit_ratio", "has_ip_host", "has_at_symbol",
    "has_double_slash_redirect", "has_https_token_in_domain", "is_shortened",
    "has_suspicious_word", "suspicious_word_count", "has_suspicious_tld",
    "has_port", "uses_https", "hostname_has_digits", "dns_resolves", "has_valid_tls",
]


def extract_all_features(url, live_checks=True):
    """Returns (feature_dict, hostname, scheme) using FEATURE_ORDER as the schema."""
    lex, hostname, scheme = extract_lexical_features(url)
    host = extract_host_features(hostname) if live_checks else {"dns_resolves": 0, "has_valid_tls": 0}
    lex.update(host)
    ordered = {k: lex[k] for k in FEATURE_ORDER}
    return ordered, hostname, scheme


def explain_features(feats):
    """Human-readable list of risk signals triggered by this URL, for the UI."""
    reasons = []
    if feats.get("has_ip_host"):
        reasons.append("Hostname is a raw IP address instead of a domain name.")
    if feats.get("has_at_symbol"):
        reasons.append("URL contains an '@' symbol, often used to obscure the real destination.")
    if feats.get("is_shortened"):
        reasons.append("URL uses a link-shortening service, hiding the true destination.")
    if feats.get("has_suspicious_word"):
        reasons.append(f"Contains {feats.get('suspicious_word_count')} suspicious keyword(s) (e.g. 'verify', 'login', 'secure').")
    if feats.get("has_suspicious_tld"):
        reasons.append("Uses a top-level domain frequently associated with phishing (e.g. .tk, .xyz, .top).")
    if feats.get("has_https_token_in_domain"):
        reasons.append("The word 'https' appears inside the domain/hostname itself (a common spoofing trick).")
    if feats.get("num_subdomains", 0) >= 3:
        reasons.append("Unusually high number of subdomains.")
    if feats.get("url_length", 0) > 75:
        reasons.append("URL is unusually long.")
    if not feats.get("uses_https"):
        reasons.append("Connection is not secured with HTTPS.")
    if feats.get("has_double_slash_redirect"):
        reasons.append("Contains a '//' redirect pattern within the path.")
    if feats.get("dns_resolves") == 0:
        reasons.append("Domain does not currently resolve via DNS (or could not be checked).")
    if not reasons:
        reasons.append("No strong lexical risk indicators detected.")
    return reasons
