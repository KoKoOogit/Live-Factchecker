from duckduckgo_search import DDGS

dgs = DDGS()
def search_internet(query):
    result = dgs.text(query)
    return query