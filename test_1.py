import betfairlightweight

api = betfairlightweight.APIClient('smudge2049', 'Nocommsnobombs17!', '4oAYsDJiYA7P5Wej')
api.login_interactive()

market_books = api.betting.list_market_book(market_ids=['1.252014561'], price_projection={'priceData': ['EX_ALL_OFFERS']})

print(market_books[0].runners[0].ex.available_to_back[0].price)

