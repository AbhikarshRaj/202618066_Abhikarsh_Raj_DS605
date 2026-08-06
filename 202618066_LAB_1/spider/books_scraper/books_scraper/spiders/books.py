import scrapy
class BooksSpider(scrapy.Spider):
    name = "books"
    allowed_domains = ["books.toscrape.com"]
    start_urls = ["https://books.toscrape.com"]
    RATING_MAP = {
        "One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5
    }
    max_books=200
    books_queued=0
    def parse(self, response):
       book_links = response.css("article.product_pod h3 a::attr(href)").getall()
       for link in book_links:
            if self.books_queued >= self.max_books:
                return
            self.books_queued += 1
            book_url = response.urljoin(link)
            yield scrapy.Request(book_url, callback=self.parse_book)
       if self.books_queued < self.max_books:
            next_page = response.css("li.next a::attr(href)").get()
            if next_page is not None:
                next_url = response.urljoin(next_page)
                yield scrapy.Request(next_url, callback=self.parse)
    def parse_book(self, response):
        title = response.css("div.product_main h1::text").get()
        category = response.css("ul.breadcrumb li:nth-child(3) a::text").get()
        price = response.css("p.price_color::text").get()
        rating_class = response.css("p.star-rating::attr(class)").get()
        rating_word = rating_class.split()[-1] if rating_class else None
        rating = self.RATING_MAP.get(rating_word)
        availability_text = response.css("p.instock.availability::text").getall()
        availability = " ".join(t.strip() for t in availability_text if t.strip())
        description = response.css("#product_description ~ p::text").get()
        table_headers = response.css("table.table.table-striped th::text").getall()
        table_values = response.css("table.table.table-striped td::text").getall()
        table_data = dict(zip(table_headers, table_values))
        yield {
            "title": title,
            "category": category,
            "price": price,
            "rating": rating,
            "availability": availability,
            "description": description,
            "upc": table_data.get("UPC"),
            "num_reviews": table_data.get("Number of reviews"),
            "product_url": response.url,
        }