import numpy as np
import pandas as pd 



def preprocess_dataset(review_csv, business_csv):
    #read reviews and business dataset
    print("=========Starting Preprocessing=========")
    review_df = pd.read_csv(review_csv)
    business_df = pd.read_csv(business_csv)

    print("reviews dataset:", review_df.columns )
    print("=========Reading Review CSV=========")
    print(review_df.head())
    """
    Expected
    reviews dataset: Index(['review_id', 'user_id', 'business_id', 'stars', 'date', 'text',
        'useful', 'funny', 'cool'],
        dtype='object')
                    review_id                 user_id             business_id  \
    0  vkVSCC7xljjrAI4UGfnKEQ  bv2nCi5Qv5vroFiqKGopiw  AEx2SYEUJmTxVVB18LlCwA   
    1  n6QzIUObkYshz4dz2QRJTw  bv2nCi5Qv5vroFiqKGopiw  VR6GpWIda3SfvPC-lg9H3w   
    2  MV3CcKScW05u5LVfF6ok0g  bv2nCi5Qv5vroFiqKGopiw  CKC0-MOWMqoeWf6s-szl8g   
    3  IXvOzsEMYtiJI0CARmj77Q  bv2nCi5Qv5vroFiqKGopiw  ACFtxLv8pGrrxMm6EgjreA   
    4  L_9BTb55X0GDtThi6GlZ6w  bv2nCi5Qv5vroFiqKGopiw  s2I_Ni76bjJNK9yG60iD-Q   

    stars        date                                               text  \
    0      5  2016-05-28  Super simple place but amazing nonetheless. It...   
    1      5  2016-05-28  Small unassuming place that changes their menu...   
    2      5  2016-05-28  Lester's is located in a beautiful neighborhoo...   
    3      4  2016-05-28  Love coming here. Yes the place always needs t...   
    4      4  2016-05-28  Had their chocolate almond croissant and it wa...   

    useful  funny  cool  
    0       0      0     0  
    1       0      0     0  
    2       0      0     0  
    3       0      0     0  
    4       0      0     0  
    """

    print("business dataset:", business_df.columns)
    print("=========Reading Business CSV=========")
    print(business_df.head())

    """
    Output exceeds the size limit. Open the full output data in a text editor
    business dataset: Index(['business_id', 'name', 'neighborhood', 'address', 'city', 'state',
        'postal_code', 'latitude', 'longitude', 'stars', 'review_count',
        'is_open', 'categories'],
        dtype='object')
                business_id                        name neighborhood  \
    0  FYWN1wneV18bWNgQjJ2GNg          "Dental by Design"          NaN   
    1  He-G7vWjzVUysIKrfNbPUQ       "Stephen Szabo Salon"          NaN   
    2  KQPW8lFf1y5BT2MxiSZ3QA     "Western Motor Vehicle"          NaN   
    3  8DShNS-LuFqpEWIp0HxijA          "Sports Authority"          NaN   
    4  PfOCPjBrlQAnz__NXj9h_w  "Brick House Tavern + Tap"          NaN   

                                address            city state postal_code  \
    0        "4855 E Warner Rd, Ste B9"       Ahwatukee    AZ       85044   
    1              "3101 Washington Rd"        McMurray    PA       15317   
    2          "6025 N 27th Ave, Ste 1"         Phoenix    AZ       85017   
    3  "5000 Arizona Mills Cr, Ste 435"           Tempe    AZ       85282   
    4                    "581 Howe Ave"  Cuyahoga Falls    OH       44221   

        latitude   longitude  stars  review_count  is_open  \
    0  33.330690 -111.978599    4.0            22        1   
    1  40.291685  -80.104900    3.0            11        1   
    2  33.524903 -112.115310    1.5            18        1   
    3  33.383147 -111.964725    3.0             9        0   
    4  41.119535  -81.475690    3.5           116        1   
    ...
    1  Hair Stylists;Hair Salons;Men's Hair Salons;Bl...  
    2  Departments of Motor Vehicles;Public Services ...  
    3                            Sporting Goods;Shopping  
    4  American (New);Nightlife;Bars;Sandwiches;Ameri...  

    """

    #Fetch the  name of the business

    bsn_name_df = business_df[['business_id', 'name']]
    print(bsn_name_df.head(5))

    """
                business_id                        name
    0  FYWN1wneV18bWNgQjJ2GNg          "Dental by Design"
    1  He-G7vWjzVUysIKrfNbPUQ       "Stephen Szabo Salon"
    2  KQPW8lFf1y5BT2MxiSZ3QA     "Western Motor Vehicle"
    3  8DShNS-LuFqpEWIp0HxijA          "Sports Authority"
    4  PfOCPjBrlQAnz__NXj9h_w  "Brick House Tavern + Tap"
    """
    print("=========Merging Business and Review CSV=========")

    #since we have business_id(unique) in both the df's, merge it
    yelp_review_df = pd.merge(review_df, bsn_name_df, how = 'left', left_on = 'business_id', right_on = 'business_id')
    yelp_review_df.head()

    """
        review_id	user_id	business_id	stars	date	text	useful	funny	cool	name
    0	vkVSCC7xljjrAI4UGfnKEQ	bv2nCi5Qv5vroFiqKGopiw	AEx2SYEUJmTxVVB18LlCwA	5	2016-05-28	Super simple place but amazing nonetheless. It...	0	0	0	"Wilensky's"
    1	n6QzIUObkYshz4dz2QRJTw	bv2nCi5Qv5vroFiqKGopiw	VR6GpWIda3SfvPC-lg9H3w	5	2016-05-28	Small unassuming place that changes their menu...	0	0	0	"Tuck Shop"
    2	MV3CcKScW05u5LVfF6ok0g	bv2nCi5Qv5vroFiqKGopiw	CKC0-MOWMqoeWf6s-szl8g	5	2016-05-28	Lester's is located in a beautiful neighborhoo...	0	0	0	"Lester's Deli"
    3	IXvOzsEMYtiJI0CARmj77Q	bv2nCi5Qv5vroFiqKGopiw	ACFtxLv8pGrrxMm6EgjreA	4	2016-05-28	Love coming here. Yes the place always needs t...	0	0	0	"Five Guys"
    4	L_9BTb55X0GDtThi6GlZ6w	bv2nCi5Qv5vroFiqKGopiw	s2I_Ni76bjJNK9yG60iD-Q	4	2016-05-28	Had their chocolate almond croissant and it wa...	0	0	0	"Maison Christian Faure
    """
    print("=========Storing Business_reviews.csv merged=========")
    #Stroing in to csv
    yelp_review_df.to_csv('../Dataset/business_reviews.csv', encoding='utf-8')

def main():
    print("=========Yelp Business Review Analysis=========")
    preprocess_dataset(
        review_csv = "../Dataset/yelp_review.csv", 
        business_csv = "../Dataset/yelp_business.csv"
        )
    print("=========Preprocessing Complete=========")

if __name__ == "__main__":
    main()