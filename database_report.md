# 📊 Database Report

📅 **Generated on:** 2025-03-21 15:34:01

## 📌 List of Tables

- stock_info
- news_sentiment
- stocks
- technical_indicators
- processed_stock_data
- latest_stock_data
- high_momentum_stocks
- oversold_stocks
- market_trend_analysis
- stock_sector_analysis

## 📊 Schema of `stock_info`

| Column Name | Data Type | Nullable | Default Value |
|------------|----------|----------|--------------|
| ticker | character varying | NO | NULL |
| company_name | text | YES | NULL |
| sector | text | YES | NULL |
| industry | text | YES | NULL |
| exchange | text | YES | NULL |
| market_cap | bigint | YES | NULL |
| pe_ratio | numeric | YES | NULL |
| eps | numeric | YES | NULL |
| earnings_date | date | YES | NULL |
| ipo_date | date | YES | NULL |
| price_to_sales_ratio | numeric | YES | NULL |
| price_to_book_ratio | numeric | YES | NULL |
| enterprise_value | bigint | YES | NULL |
| ebitda | bigint | YES | NULL |
| profit_margin | numeric | YES | NULL |
| return_on_equity | numeric | YES | NULL |
| beta | numeric | YES | NULL |
| dividend_yield | numeric | YES | NULL |

## 📊 Schema of `news_sentiment`

| Column Name | Data Type | Nullable | Default Value |
|------------|----------|----------|--------------|
| id | integer | NO | nextval('news_sentiment_id_seq'::regclass) |
| ticker | character varying | NO | NULL |
| published_at | timestamp without time zone | NO | NULL |
| source_name | text | YES | NULL |
| title | text | NO | NULL |
| description | text | YES | NULL |
| url | text | NO | NULL |
| sentiment_score | numeric | YES | NULL |
| created_at | timestamp without time zone | YES | now() |

## 📊 Schema of `stocks`

| Column Name | Data Type | Nullable | Default Value |
|------------|----------|----------|--------------|
| id | integer | NO | nextval('stocks_id_seq'::regclass) |
| ticker | character varying | NO | NULL |
| date | timestamp without time zone | NO | NULL |
| open | numeric | YES | NULL |
| high | numeric | YES | NULL |
| low | numeric | YES | NULL |
| close | numeric | YES | NULL |
| volume | bigint | YES | NULL |
| adjusted_close | numeric | YES | NULL |

## 📊 Schema of `technical_indicators`

| Column Name | Data Type | Nullable | Default Value |
|------------|----------|----------|--------------|
| id | integer | NO | nextval('technical_indicators_id_seq'::regclass) |
| ticker | character varying | NO | NULL |
| date | timestamp without time zone | NO | NULL |
| sma_50 | numeric | YES | NULL |
| sma_200 | numeric | YES | NULL |
| ema_50 | numeric | YES | NULL |
| ema_200 | numeric | YES | NULL |
| rsi_14 | numeric | YES | NULL |
| adx_14 | numeric | YES | NULL |
| atr_14 | numeric | YES | NULL |
| cci_20 | numeric | YES | NULL |
| williamsr_14 | numeric | YES | NULL |
| macd | numeric | YES | NULL |
| macd_signal | numeric | YES | NULL |
| macd_hist | numeric | YES | NULL |
| bb_upper | numeric | YES | NULL |
| bb_middle | numeric | YES | NULL |
| bb_lower | numeric | YES | NULL |
| stoch_k | numeric | YES | NULL |
| stoch_d | numeric | YES | NULL |

## 📊 Schema of `processed_stock_data`

| Column Name | Data Type | Nullable | Default Value |
|------------|----------|----------|--------------|
| ticker | character varying | NO | NULL |
| date | timestamp without time zone | NO | NULL |
| open | double precision | YES | NULL |
| high | double precision | YES | NULL |
| low | double precision | YES | NULL |
| close | double precision | YES | NULL |
| volume | double precision | YES | NULL |
| adjusted_close | double precision | YES | NULL |
| sma_50 | double precision | YES | NULL |
| sma_200 | double precision | YES | NULL |
| ema_50 | double precision | YES | NULL |
| ema_200 | double precision | YES | NULL |
| rsi_14 | double precision | YES | NULL |
| adx_14 | double precision | YES | NULL |
| atr_14 | double precision | YES | NULL |
| cci_20 | double precision | YES | NULL |
| williamsr_14 | double precision | YES | NULL |
| macd | double precision | YES | NULL |
| macd_signal | double precision | YES | NULL |
| macd_hist | double precision | YES | NULL |
| bb_upper | double precision | YES | NULL |
| bb_lower | double precision | YES | NULL |
| stoch_k | double precision | YES | NULL |
| stoch_d | double precision | YES | NULL |
| sentiment_score | double precision | YES | NULL |
| returns | double precision | YES | NULL |
| volatility | double precision | YES | NULL |
| close_lag_1 | double precision | YES | NULL |
| volume_lag_1 | double precision | YES | NULL |
| close_lag_5 | double precision | YES | NULL |
| volume_lag_5 | double precision | YES | NULL |
| close_lag_10 | double precision | YES | NULL |
| volume_lag_10 | double precision | YES | NULL |

## 📊 Schema of `latest_stock_data`

| Column Name | Data Type | Nullable | Default Value |
|------------|----------|----------|--------------|
| id | integer | YES | NULL |
| ticker | character varying | YES | NULL |
| date | timestamp without time zone | YES | NULL |
| open | numeric | YES | NULL |
| high | numeric | YES | NULL |
| low | numeric | YES | NULL |
| close | numeric | YES | NULL |
| volume | bigint | YES | NULL |
| adjusted_close | numeric | YES | NULL |

## 📊 Schema of `high_momentum_stocks`

| Column Name | Data Type | Nullable | Default Value |
|------------|----------|----------|--------------|
| ticker | character varying | YES | NULL |
| date | timestamp without time zone | YES | NULL |
| close | numeric | YES | NULL |
| rsi_14 | numeric | YES | NULL |
| macd | numeric | YES | NULL |
| macd_signal | numeric | YES | NULL |

## 📊 Schema of `oversold_stocks`

| Column Name | Data Type | Nullable | Default Value |
|------------|----------|----------|--------------|
| ticker | character varying | YES | NULL |
| date | timestamp without time zone | YES | NULL |
| close | numeric | YES | NULL |
| rsi_14 | numeric | YES | NULL |

## 📊 Schema of `market_trend_analysis`

| Column Name | Data Type | Nullable | Default Value |
|------------|----------|----------|--------------|
| ticker | character varying | YES | NULL |
| min_price | numeric | YES | NULL |
| max_price | numeric | YES | NULL |
| avg_price | numeric | YES | NULL |
| days_tracked | bigint | YES | NULL |

## 📊 Schema of `stock_sector_analysis`

| Column Name | Data Type | Nullable | Default Value |
|------------|----------|----------|--------------|
| sector | text | YES | NULL |
| total_stocks | bigint | YES | NULL |
| avg_sector_price | numeric | YES | NULL |

## 🔍 Sample Data from `stock_info`

| ticker | company_name | sector | industry | exchange | market_cap | pe_ratio | eps | earnings_date | ipo_date | price_to_sales_ratio | price_to_book_ratio | enterprise_value | ebitda | profit_margin | return_on_equity | beta | dividend_yield |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| IMVT | Immunovant, Inc. | Healthcare | Biotechnology | NMS | 3289358080 | NULL | -2.62 | NULL | NULL | NULL | 8.085595 | 2969894912 | -404873984 | NULL | -0.74170995 | 0.676 | NULL |
| AAL | American Airlines Group Inc. | Industrials | Airlines | NMS | 7424032768 | 9.104838 | 1.24 | NULL | NULL | 0.13694699 | -1.8667328 | 37330300928 | 5615000064 | 0.01561 | NULL | 1.27 | NULL |
| AAPL | Apple Inc. | Technology | Consumer Electronics | NMS | 3231028609024 | 34.194756 | 6.29 | NULL | NULL | 8.164111 | 48.464397 | 3259249721344 | 137352003584 | 0.24295 | 1.3652 | 1.178 | 0.47 |
| ACHR | Archer Aviation Inc. | Industrials | Aerospace & Defense | NYQ | 4723503616 | NULL | -1.42 | NULL | NULL | NULL | 5.8282466 | 3736153856 | -498000000 | NULL | -0.95883006 | 3.143 | NULL |
| ACN | Accenture plc | Technology | Information Technology Services | NYQ | 190044717056 | 25.009068 | 12.13 | NULL | NULL | 2.8271422 | 6.497879 | 188878159872 | 11472501760 | 0.114300005 | 0.26965 | 1.239 | 1.97 |

## 🔍 Sample Data from `news_sentiment`

| id | ticker | published_at | source_name | title | description | url | sentiment_score | created_at |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | META | 2025-03-19 14:41:00 | Google News | META Stock: Why I'm Betting Big Before The Surge (NASDAQ:META) - Seeking Alpha | <a href='https://news.google.com/rss/articles/CBMikAFBVV95cUxPenZFdW5oNGZQWnRvZS1VTFVwamJ5N29qLS1IenYwSHM0SjRmWVoyS2hOTjRYbjh1RndoQmZPQnY1U1JidE10UHBZSHJoekRkSVNITlpMN2k5bHVHTW5uSlByenZPY2hWY2RHWlY5T1ZoVWpubFdDSEZJbmxWcnh1YWd3a2VMMUVFRzhfN2syelA?oc=5' target='_blank'>META Stock: Why I'm Betting Big Before The Surge (NASDAQ:META)</a>&nbsp;&nbsp;<font color='#6f6f6f'>Seeking Alpha</font> | https://news.google.com/rss/articles/CBMikAFBVV95cUxPenZFdW5oNGZQWnRvZS1VTFVwamJ5N29qLS1IenYwSHM0SjRmWVoyS2hOTjRYbjh1RndoQmZPQnY1U1JidE10UHBZSHJoekRkSVNITlpMN2k5bHVHTW5uSlByenZPY2hWY2RHWlY5T1ZoVWpubFdDSEZJbmxWcnh1YWd3a2VMMUVFRzhfN2syelA?oc=5 | NULL | 2025-03-20 17:01:46 |
| 2 | META | 2025-03-20 08:31:00 | Google News | Market Sell-Off: Should Investors Buy the Dip in Meta Platforms Stock? - The Motley Fool | <a href='https://news.google.com/rss/articles/CBMimAFBVV95cUxPQ2ttcTFCelhIaWhNa1NDUi1fNHc5NFpBRVlSNDlEX0FUa3lUNk1OY3ZfMDBJV1VwSGQxejM1aXhvVk5Ba0ZuZzY2NkpxdC1NczdacmdPNGJicHBOYXNNcXFCXy14Zjk1eFNrNHJwckR2WVRkeHhkQWktY21qRmdXd2JPVkZ0cDJQRWxIUHJtSHRrTEg5ZENaYg?oc=5' target='_blank'>Market Sell-Off: Should Investors Buy the Dip in Meta Platforms Stock?</a>&nbsp;&nbsp;<font color='#6f6f6f'>The Motley Fool</font> | https://news.google.com/rss/articles/CBMimAFBVV95cUxPQ2ttcTFCelhIaWhNa1NDUi1fNHc5NFpBRVlSNDlEX0FUa3lUNk1OY3ZfMDBJV1VwSGQxejM1aXhvVk5Ba0ZuZzY2NkpxdC1NczdacmdPNGJicHBOYXNNcXFCXy14Zjk1eFNrNHJwckR2WVRkeHhkQWktY21qRmdXd2JPVkZ0cDJQRWxIUHJtSHRrTEg5ZENaYg?oc=5 | -0.533 | 2025-03-20 17:01:46 |
| 3 | META | 2025-03-18 21:11:00 | Google News | Meta Stock Slides To Approach 3-Month Low Despite This AI Milestone - Investor's Business Daily | <a href='https://news.google.com/rss/articles/CBMirwFBVV95cUxOSUwtVE0ySFd1NnJ4UjVjdTR2cFctN25INFlOeWNoOVY1anZWbmRIcEt0aWh6QndnZ05WYnNrWEFOZU93STZNU0dqcGNROE5yQkhEci1yRHBNVkRocWtkQUxpazNQd2N1S0UwbU9yMkJmTVZVR3BkWlQtZ0xhMTZVS3hXaUpfS1hfZlgtdjdPYk95cXI0dV9ubGx2NTl3MzB2OTZmMzBtdm5walpzT0lB?oc=5' target='_blank'>Meta Stock Slides To Approach 3-Month Low Despite This AI Milestone</a>&nbsp;&nbsp;<font color='#6f6f6f'>Investor's Business Daily</font> | https://news.google.com/rss/articles/CBMirwFBVV95cUxOSUwtVE0ySFd1NnJ4UjVjdTR2cFctN25INFlOeWNoOVY1anZWbmRIcEt0aWh6QndnZ05WYnNrWEFOZU93STZNU0dqcGNROE5yQkhEci1yRHBNVkRocWtkQUxpazNQd2N1S0UwbU9yMkJmTVZVR3BkWlQtZ0xhMTZVS3hXaUpfS1hfZlgtdjdPYk95cXI0dV9ubGx2NTl3MzB2OTZmMzBtdm5walpzT0lB?oc=5 | -0.4939 | 2025-03-20 17:01:46 |
| 4 | META | 2025-03-19 17:39:10 | Google News | Meta Platforms Stock (NASDAQ:META) Slips as it Considers Leaving Delaware - The Globe and Mail | <a href='https://news.google.com/rss/articles/CBMi7AFBVV95cUxPMjVnaWVERERPLWRFT1lJYlJmSW9vRWliSWFJeVJCTHVuMkJIX2RUQTN6MmF3akc2YjdGTHVPdl84WTNfY0dJSWZZSUF5Mk1CeUYwV1BEUlhnMmNHX0JBZE5vbjRwTDNhdmJERkY1TW8zOWZJQmVzZ3FTMXhjOE5uQjVTTlpqLWlpMW5IQ2NaWjdKejNSb1dKSVI5bFVyR194dFFaV2NieEpQNjdtM3oyMEFwMGhOMTdadWs0MVVUeWtIdm5OTjlLdml3RFdUM3JsZlpnXzdWY01ZY3RKcFhlZVhUZjVpSFVlc0R2WQ?oc=5' target='_blank'>Meta Platforms Stock (NASDAQ:META) Slips as it Considers Leaving Delaware</a>&nbsp;&nbsp;<font color='#6f6f6f'>The Globe and Mail</font> | https://news.google.com/rss/articles/CBMi7AFBVV95cUxPMjVnaWVERERPLWRFT1lJYlJmSW9vRWliSWFJeVJCTHVuMkJIX2RUQTN6MmF3akc2YjdGTHVPdl84WTNfY0dJSWZZSUF5Mk1CeUYwV1BEUlhnMmNHX0JBZE5vbjRwTDNhdmJERkY1TW8zOWZJQmVzZ3FTMXhjOE5uQjVTTlpqLWlpMW5IQ2NaWjdKejNSb1dKSVI5bFVyR194dFFaV2NieEpQNjdtM3oyMEFwMGhOMTdadWs0MVVUeWtIdm5OTjlLdml3RFdUM3JsZlpnXzdWY01ZY3RKcFhlZVhUZjVpSFVlc0R2WQ?oc=5 | NULL | 2025-03-20 17:01:46 |
| 5 | META | 2025-03-19 13:07:40 | Google News | Fed decision, Nvidia unveils chips, Meta declines: 3 Things - Yahoo Finance | <a href='https://news.google.com/rss/articles/CBMihwFBVV95cUxORGV2NzM4VDIxaExaQ0hoOHhQV0xJR3ZTZDViMVRaY04zdG5aQ25kNU5Rc1htTDd1UWc4Y25JaGRlVEg3emtnQThhZklXdjY3NWxZeGpMY05pbjdISHJyZXRheFJEZmd6MnZ5YWM2TTNwZ3YxcUpmY0JmWUxyd3RmbjU4cC1TUUE?oc=5' target='_blank'>Fed decision, Nvidia unveils chips, Meta declines: 3 Things</a>&nbsp;&nbsp;<font color='#6f6f6f'>Yahoo Finance</font> | https://news.google.com/rss/articles/CBMihwFBVV95cUxORGV2NzM4VDIxaExaQ0hoOHhQV0xJR3ZTZDViMVRaY04zdG5aQ25kNU5Rc1htTDd1UWc4Y25JaGRlVEg3emtnQThhZklXdjY3NWxZeGpMY05pbjdISHJyZXRheFJEZmd6MnZ5YWM2TTNwZ3YxcUpmY0JmWUxyd3RmbjU4cC1TUUE?oc=5 | NULL | 2025-03-20 17:01:46 |

## 🔍 Sample Data from `stocks`

| id | ticker | date | open | high | low | close | volume | adjusted_close |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 258830 | CPRX | 2023-05-01 00:00:00 | 15.9399995803833 | 16.959999084472656 | 15.90999984741211 | 16.940000534057617 | 1594900 | 16.940000534057617 |
| 258831 | CPRX | 2023-05-02 00:00:00 | 16.920000076293945 | 17.25 | 16.59000015258789 | 16.6200008392334 | 1683200 | 16.6200008392334 |
| 258832 | CPRX | 2023-05-03 00:00:00 | 16.6299991607666 | 16.81999969482422 | 16.350000381469727 | 16.600000381469727 | 1550300 | 16.600000381469727 |
| 258833 | CPRX | 2023-05-04 00:00:00 | 16.579999923706055 | 16.770000457763672 | 16.25 | 16.65999984741211 | 1434500 | 16.65999984741211 |
| 258834 | CPRX | 2023-05-05 00:00:00 | 16.790000915527344 | 17.540000915527344 | 16.790000915527344 | 17.209999084472656 | 1859800 | 17.209999084472656 |

## 🔍 Sample Data from `technical_indicators`

| id | ticker | date | sma_50 | sma_200 | ema_50 | ema_200 | rsi_14 | adx_14 | atr_14 | cci_20 | williamsr_14 | macd | macd_signal | macd_hist | bb_upper | bb_middle | bb_lower | stoch_k | stoch_d |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2359 | RKT | 2021-05-21 00:00:00 | 20.056532745361327 | 19.65207783699036 | 19.10927253103879 | 19.65207783699036 | 30.022311807720165 | 32.788243513099246 | 0.8869549675399673 | -502.8154833676608 | -89.87339153755882 | -1.3517839014553594 | -1.2181714954618283 | -0.133612405993531 | 23.019910719802663 | 18.14191884994507 | 13.26392698008748 | 12.18941800069036 | 15.080498663548765 |
| 2360 | RKT | 2021-05-24 00:00:00 | 19.905853271484375 | 19.635083379745485 | 18.97556176132913 | 19.61275017596003 | 29.837350308557124 | 32.954878237959626 | 0.844768376989419 | -517.3750173514725 | -90.43599433357105 | -1.3322033424841635 | -1.2409778648662957 | -0.091225477617868 | 22.686274353144185 | 17.878554916381837 | 13.07083547961949 | 11.29865310156596 | 12.93953581047934 |
| 2361 | RKT | 2021-05-25 00:00:00 | 19.73882387161255 | 19.6026211309433 | 18.84381581891612 | 19.57298191778926 | 29.398497983688745 | 33.164679833709066 | 0.8139558374305087 | -544.3008400103002 | -83.51953444092635 | -1.3083502022421278 | -1.254452332341462 | -0.0538978699006658 | 22.233869803261335 | 17.595682573318484 | 12.957495343375632 | 12.05702656264793 | 11.848365888301416 |
| 2362 | RKT | 2021-05-26 00:00:00 | 19.59873468399048 | 19.58751747608185 | 18.7409160425271 | 19.539617635234105 | 36.64597011587443 | 32.61408019717566 | 0.8075732595608192 | -595.5831185636638 | -58.389304744545335 | -1.2265829927575318 | -1.248878464424676 | 0.0222954716671441 | 21.702478337608163 | 17.344859647750855 | 12.987240957893547 | 22.551722160319095 | 15.302467274844329 |
| 2363 | RKT | 2021-05-27 00:00:00 | 19.48447095870972 | 19.58138677120209 | 18.67447460829484 | 19.514812081347245 | 44.975024162095195 | 30.693742018369317 | 0.8925531789147222 | -690.715241847197 | -33.018838368518374 | -1.0825876162893593 | -1.215620294797613 | 0.1330326785082534 | 21.10102905813846 | 17.133518266677857 | 13.166007475217253 | 41.69077414866997 | 25.433174290545665 |

## 🔍 Sample Data from `processed_stock_data`

| ticker | date | open | high | low | close | volume | adjusted_close | sma_50 | sma_200 | ema_50 | ema_200 | rsi_14 | adx_14 | atr_14 | cci_20 | williamsr_14 | macd | macd_signal | macd_hist | bb_upper | bb_lower | stoch_k | stoch_d | sentiment_score | returns | volatility | close_lag_1 | volume_lag_1 | close_lag_5 | volume_lag_5 | close_lag_10 | volume_lag_10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| META | 2015-10-16 00:00:00 | -1.1045735101983611 | -1.101700238895816 | -1.1049266273766432 | -1.0956749765086404 | 0.21992001286409105 | -1.0956749765086404 | -1.098056200129254 | -1.2149744791244672 | -1.1085171631963535 | -1.2297873403216633 | 0.8933658588393872 | -1.3253624650725142 | -0.9353319324107892 | -0.15637058003423898 | 1.442503825250569 | 0.0667072074217626 | -0.037167194990615667 | 0.3443228850869841 | -1.0988124381432796 | -1.0948471350965512 | 1.3131747221072165 | 1.1668143134968465 | -0.08914762402063287 | -0.04354836537585999 | -1.2716832852411981 | -1.783799792492372 | -1.5061811107438252 | -1.78119051478322 | -1.5025458132509686 | -1.7780494455643667 | -1.4980256635833609 |
| META | 2015-10-19 00:00:00 | -1.0975101899524953 | -1.094884425138012 | -1.0936509839580768 | -1.0891077330389884 | 0.1055465152272284 | -1.0891077330389884 | -1.0974779961138774 | -1.2141505368906549 | -1.1067421101847867 | -1.2286466826856508 | 1.0557855502885007 | -1.2601624232944524 | -0.9460218481449012 | -0.08441303256210168 | 1.4249173882658233 | 0.10214735396528557 | -0.014797292155195414 | 0.3992377745561706 | -1.0949467799764343 | -1.0972245399028588 | 1.5123697893789088 | 1.3557814345400692 | -0.08914762402063287 | 0.3597298656009161 | -1.2716832852411981 | -1.0945921628515727 | 0.22018355713540239 | -1.78119051478322 | -1.5025458132509686 | -1.7780494455643667 | -1.4980256635833609 |
| META | 2015-10-20 00:00:00 | -1.0865619137933515 | -1.0877903868540013 | -1.0962364918749583 | -1.099488312435168 | 0.5915761157911488 | -1.099488312435168 | -1.0970828214111632 | -1.213335237723022 | -1.1054419327848608 | -1.2276382738999374 | 0.5841673046618744 | -1.1628634494085175 | -0.9373434166310408 | 0.04464441071076587 | 0.6851446279855229 | 0.11258883513092774 | 0.005481322029506482 | 0.37381091121447313 | -1.091979429494722 | -1.097703680736959 | 1.2745337207847125 | 1.4112439062391058 | -0.08914762402063287 | -0.6749738543684642 | -1.2716832852411981 | -1.0880209413763187 | 0.10585297164375286 | -1.78119051478322 | -1.5025458132509686 | -1.7780494455643667 | -1.4980256635833609 |
| META | 2015-10-21 00:00:00 | -1.09616808593266 | -1.0975968090588784 | -1.0968828092105116 | -1.0987114957209732 | -0.017600374296098304 | -1.0987114957209732 | -1.0965989039996187 | -1.2124726088006834 | -1.1041624152204894 | -1.2266308517165097 | 0.6065209238680622 | -1.0780331160588157 | -0.9470807812221246 | 0.14248958714207016 | 0.7172516810146279 | 0.11981015519030634 | 0.023352081209236892 | 0.34453508640807845 | -1.0894625028533953 | -1.0983246638081754 | 1.014739700196383 | 1.3083975822446525 | -0.08914762402063287 | 0.004419597957632477 | -1.2716832852411981 | -1.0984078086470352 | 0.591700217263318 | -1.78119051478322 | -1.5025458132509686 | -1.7780494455643667 | -1.4980256635833609 |
| META | 2015-10-22 00:00:00 | -1.0932015313159955 | -1.0868166640144776 | -1.089126311009224 | -1.0806338216060727 | 0.2864645933073566 | -1.0806338216060727 | -1.0958390587710578 | -1.2115046210372742 | -1.1022273087125478 | -1.2254229092074995 | 1.0807043457435088 | -0.9449187880127133 | -0.94385427979745 | 0.26606983926769134 | 1.434917083531629 | 0.14998770013580434 | 0.04453504799348075 | 0.3848064521472676 | -1.0839982786903903 | -1.1008377950044954 | 1.0183217110300062 | 1.1381393182869781 | -0.08914762402063287 | 1.0714725827238354 | -1.2716832852411981 | -1.0976305213881963 | -0.017247714018408807 | -1.78119051478322 | -1.5025458132509686 | -1.7780494455643667 | -1.4980256635833609 |

## 🔍 Sample Data from `latest_stock_data`

| id | ticker | date | open | high | low | close | volume | adjusted_close |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 156869 | AAL | 2025-03-21 00:00:00 | 11.0649995803833 | 11.3100004196167 | 10.96500015258789 | 11.28499984741211 | 35418554 | 11.28499984741211 |
| 193178 | AAPL | 2025-03-21 00:00:00 | 211.51499938964844 | 215.47999572753903 | 211.47000122070312 | 214.9149932861328 | 41999242 | 214.9149932861328 |
| 284571 | ACHR | 2025-03-21 00:00:00 | 8.119999885559082 | 8.760000228881836 | 8.050000190734863 | 8.71500015258789 | 20101802 | 8.71500015258789 |
| 35454 | ACN | 2025-03-20 00:00:00 | 296.1499938964844 | 306.8500061035156 | 291.54998779296875 | 301.2900085449219 | 6737322 | 301.2900085449219 |
| 94281 | ACVA | 2025-03-20 00:00:00 | 15.140000343322754 | 15.600000381469728 | 14.890000343322754 | 15.390000343322754 | 2606666 | 15.390000343322754 |

## 🔍 Sample Data from `high_momentum_stocks`

| ticker | date | close | rsi_14 | macd | macd_signal |
| --- | --- | --- | --- | --- | --- |
| VVPR | 2016-06-17 00:00:00 | 100.6999969482422 | 98.16053124125511 | 0.1828961482967486 | 0.1525688746324813 |
| VVPR | 2016-06-16 00:00:00 | 100.6999969482422 | 98.16053124125511 | 0.1813863522571495 | 0.1449870562164144 |
| VVPR | 2016-06-15 00:00:00 | 100.6999969482422 | 98.16053124125511 | 0.1768209565519498 | 0.1358872322062307 |
| VVPR | 2016-06-14 00:00:00 | 100.6999969482422 | 98.16053124125511 | 0.1684219103020297 | 0.1256538011198009 |
| VVPR | 2016-06-13 00:00:00 | 100.6999969482422 | 98.16053124125511 | 0.155251899847741 | 0.1149617738242437 |

## 🔍 Sample Data from `oversold_stocks`

| ticker | date | close | rsi_14 |
| --- | --- | --- | --- |
| HOLO | 2022-09-09 00:00:00 | 1364.0 | 4.859162027480148 |
| QBTS | 2022-07-28 00:00:00 | 8.720000267028809 | 4.909016736737972 |
| HOLO | 2022-09-08 00:00:00 | 1480.0 | 5.846751221313623 |
| ALUR | 2023-08-03 00:00:00 | 132.25 | 6.730195880113316 |
| RGTI | 2022-02-23 00:00:00 | 8.399999618530273 | 7.24965745855682 |

## 🔍 Sample Data from `market_trend_analysis`

| ticker | min_price | max_price | avg_price | days_tracked |
| --- | --- | --- | --- | --- |
| TPL | 1104.720703125 | 1256.880126953125 | 1183.1867167154947917 | 12 |
| META | 582.3599853515625 | 639.4310913085938 | 608.0780906677246125 | 32 |
| LMT | 420.0891418457031 | 502.5802612304687 | 462.2601526896158867 | 60 |
| KNSL | 416.3101501464844 | 494.5182495117187 | 447.1298212687174525 | 60 |
| ACN | 301.2900085449219 | 398.25 | 358.8828828456038144 | 59 |

## 🔍 Sample Data from `stock_sector_analysis`

| sector | total_stocks | avg_sector_price |
| --- | --- | --- |
| Healthcare | 23255 | 15827455.1051447297970061 |
| Industrials | 19647 | 141.6456377621296982 |
| Energy | 7762 | 107.4683294301102440 |
| Consumer Defensive | 11337 | 86.6584091677524388 |
| Consumer Cyclical | 34509 | 66.8960602623066563 |

## 📅 Date Range of Stock Price Data

- **news_sentiment**: Start Date = 2009-05-16 18:58:54, End Date = 2025-03-21 19:17:59
- **stocks**: Start Date = 2015-10-16 00:00:00, End Date = 2025-03-21 00:00:00
- **technical_indicators**: Start Date = 2015-10-16 00:00:00, End Date = 2025-03-21 00:00:00
- **processed_stock_data**: Start Date = 2015-10-16 00:00:00, End Date = 2025-03-20 00:00:00
- **latest_stock_data**: Start Date = 2024-11-05 00:00:00, End Date = 2025-03-21 00:00:00
- **high_momentum_stocks**: Start Date = 2015-10-16 00:00:00, End Date = 2025-03-21 00:00:00
- **oversold_stocks**: Start Date = 2015-10-21 00:00:00, End Date = 2025-03-21 00:00:00
## 📊 Row Counts for Each Table

| Table Name | Row Count |
|------------|----------|
| stocks | 181266 |
| technical_indicators | 181266 |
| processed_stock_data | 107066 |
| news_sentiment | 17758 |
| stock_info | 101 |
