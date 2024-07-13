try:
    import pandas as pd
    from sklearn.preprocessing import StandardScaler, MinMaxScaler

except Exception as e:
    print("Error Occured:", e)

# Start of node 1 components
def preprocess_customers(customers: pd.DataFrame):
    customers.drop(['customer_city', 'customer_state'], axis=1,
                   inplace=True)
    return customers 

def preprocess_items(items: pd.DataFrame):
    items.drop(columns=['shipping_limit_date'], axis=1,  inplace=True)
    return items 

def preprocess_payment(payment: pd.DataFrame):
    payment = payment.drop(columns=['payment_sequential', 
                                    'payment_installments'], axis=1)
    return payment 

def preprocess_review(review: pd.DataFrame):
    review = review.drop(columns=['review_comment_title', 
                                  'review_comment_message',
                                  'review_creation_date', 
                                  'review_answer_timestamp'], axis=1)
    return review 

def preprocess_order(order: pd.DataFrame):
    order = order[(order['order_status']=='delivered')] # only get delivered
    order = order.copy()
    order.drop(columns=['order_purchase_timestamp', 
                        'order_approved_at',
                        'order_delivered_carrier_date'], axis=1, inplace=True)
    return order

def preprocess_product(product: pd.DataFrame):
    product = product.iloc[:, :2] # Select only 1st 2 cols
    return product

# Node 1
def preprocessing_indiv(
        customers: pd.DataFrame, 
        items: pd.DataFrame,
        payment: pd.DataFrame, 
        review: pd.DataFrame, 
        order: pd.DataFrame,
        product: pd.DataFrame
        ):
    # Preprocessing at the individual dataset level   
    customers = preprocess_customers(customers)
    items = preprocess_items(items)
    payment = preprocess_payment(payment)
    review = preprocess_review(review)
    order = preprocess_order(order)
    product = preprocess_product(product)
    
    return (customers, items, payment, review, order, product)

# Node 2
def merge_data(
        customers: pd.DataFrame, items: pd.DataFrame,
        payment: pd.DataFrame, review: pd.DataFrame, 
        order: pd.DataFrame, product: pd.DataFrame, 
        ):
        # List of DF to merge together
    merge_info = [
        (customers, 'customer_id', 'customer_id'),
        (payment, 'order_id', 'order_id'),
        (review, 'order_id', 'order_id'),
        (items, 'order_id', 'order_id'),
        (product, 'product_id', 'product_id'),
    ]

    # # Initialize merging with the order_df
    df = order

    # Loop through each DataFrame and merge
    for merge_df, left_key, right_key in merge_info:
        df = df.merge(merge_df, left_on=left_key, 
                      right_on=right_key, how='left')

    return df

# Node 3 Components
def repurchase(df: pd.DataFrame) -> pd.DataFrame:
    # Group by user_id and count the number of unique order_numbers for each user
    order_counts = df.groupby('customer_unique_id')['order_id'].nunique()
    mo_customer = order_counts[order_counts > 1].index
    so_customer = order_counts[order_counts == 1].index
    # Create a dictionary to map customer IDs to their respective labels
    customer_type = {customer_id: 1 for customer_id in mo_customer}
    customer_type.update({customer_id: 0 for customer_id in so_customer})

    # Map to new column indicating repurchase or not
    df['repurchase'] = df['customer_unique_id'].map(customer_type)

    return df

def order_engineering(df: pd.DataFrame) -> pd.DataFrame:
    # We follow exactly as notebook for convenience
    # Group by order_id and calculate the number of payments, unique payment methods, and sum of payment values
    result = df.groupby('order_id').agg(
        number_of_payments=('payment_type', 'count'),
        payment_methods=('payment_type', 'nunique'),
        total_payment_value=('payment_value', 'sum'),
        no_of_categories=('product_category_name', 'nunique'),
        sellers=('seller_id','nunique')
    ).reset_index()

    # Additional calculation for items bought and max order item id within product categories
    basket_info = df.groupby(['order_id', 'product_category_name']).agg(
        max_order_item_id=('order_item_id', 'max')
    ).reset_index()

    # Sum up the max order item ids to get the total number of items per order
    total_items = basket_info.groupby('order_id')['max_order_item_id'] \
                            .sum().reset_index()
    total_items.rename(columns
                    = {'max_order_item_id': 'total_items'}, inplace=True)

    # Merge the total items back into the result
    result = result.merge(total_items, on='order_id')

    # Merge the result into the actual dataset
    temp_df = df.drop_duplicates(subset=['order_id'])
    final_df = temp_df.merge(result, left_on='order_id', right_on='order_id', 
                            how='left')
    return final_df

def time_engineering(df: pd.DataFrame) -> pd.DataFrame:
    # Convert columns to datetime, coerce errors to NaT
    df['order_delivered_customer_date'] = pd.to_datetime \
                                                (df \
                                                ['order_delivered_customer_date'], 
                                                errors='coerce')
    df['order_estimated_delivery_date'] = pd.to_datetime \
                                                (df \
                                                ['order_estimated_delivery_date'], 
                                                errors='coerce')

    # Drop rows with NaT values
    df = df.dropna()
    df = df.copy()
    # Modified to not group by early late or on time
    # Convert to date only (exclude time)
    df['delivered_date'] = df['order_delivered_customer_date'].dt.date
    df['estimated_date'] = df['order_estimated_delivery_date'].dt.date
    df['days'] = (df['delivered_date'] 
                  - df['estimated_date']).apply(lambda x: x.days)
    
    return df

def value_check(df: pd.DataFrame) -> pd.DataFrame:
    df = df[(df['price'] + df['freight_value']) == df['total_payment_value']]
    return df

def drop_cols(df: pd.DataFrame) -> pd.DataFrame:
    col = ['order_id', 'customer_id', 'order_status',
    'order_delivered_customer_date', 'order_estimated_delivery_date',
    'customer_unique_id', 'customer_zip_code_prefix', 'payment_type',
    'payment_value', 'review_id', 'order_item_id', 'product_id', 'seller_id', 
    'product_category_name', 'total_payment_value', 'no_of_categories', 
    'sellers', 'total_items', 'delivered_date', 'estimated_date']
    df.drop(columns = col, axis=1, inplace=True)
    return df

# Node 3
def preprocess_all(df: pd.DataFrame) -> pd.DataFrame:
    df = repurchase(df) # Step 1
    df = order_engineering(df) # Step 2
    df = df[df['order_status']=='delivered'] # Step 3
    df = time_engineering(df) # Step 4
    df = value_check(df) # Step 5
    df = drop_cols(df) # Step 6
    return df


# Node 4
def data_processing(df: pd.DataFrame) -> pd.DataFrame:
    mms = MinMaxScaler()
    df[['price', 'freight_value']] = mms.fit_transform(df[['price', 
                                                            'freight_value']])
    std = StandardScaler()
    df[['days']] = std.fit_transform(df[['days']])

    return df




