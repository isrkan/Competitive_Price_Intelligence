from flask import Flask, render_template, request
import identifying_competitors
import imputation_forecasting
import time

app = Flask(__name__, static_url_path='/static')

@app.route('/', methods=['GET', 'POST'])
def form():
    if request.method == 'POST':
        SubChainName = request.form.get('SubChainName')
        StoreName = request.form.get('StoreName')
        category = request.form.get('category')
        product_description = request.form.get('product_description')
        Geographic = request.form.get('Geographic')
        # Process the form data and perform necessary operations
        
        # Call the identify_competitors function with user inputs
        fig1, fig2, fig3, top_competitors_df = identifying_competitors.identify_competitors(category,product_description,SubChainName,StoreName,Geographic)

        # Call the imputation_forecasting function with user inputs
        competitor_product_df, competitor_product_imputed_df, competitor_product_imputed_forecast_df = imputation_forecasting.imputation_forecasting(category, top_competitors_df, product_description)

        # Generate a timestamp
        timestamp = int(time.time())

        return render_template('identifying_competitors_results.html', fig1=fig1, fig2=fig2, fig3=fig3, top_competitors_df=top_competitors_df, timestamp=timestamp)
    
    return render_template('form.html')


if __name__ == '__main__':
    app.run(debug=True)