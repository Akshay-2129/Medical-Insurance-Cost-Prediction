import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

def generate_medical_insurance_dataset(n_samples=2000):
    """
    Generate a realistic medical insurance dataset
    """
    np.random.seed(42)
    random.seed(42)
    
    # Define possible values for categorical variables
    regions = ['northeast', 'northwest', 'southeast', 'southwest']
    sexes = ['male', 'female']
    
    data = []
    
    for i in range(n_samples):
        # Generate age (18-64, with higher probability for middle ages)
        age = int(np.random.normal(40, 12))
        age = max(18, min(64, age))
        
        # Generate sex
        sex = random.choice(sexes)
        
        # Generate BMI (18-45, normal distribution around 26)
        bmi = round(np.random.normal(26, 4), 1)
        bmi = max(18.0, min(45.0, bmi))
        
        # Generate number of children (0-5, with higher probability for 0-2)
        children = np.random.choice([0, 1, 2, 3, 4, 5], p=[0.3, 0.25, 0.25, 0.12, 0.05, 0.03])
        
        # Generate smoker status (about 20% smokers)
        smoker = 'yes' if random.random() < 0.2 else 'no'
        
        # Generate region
        region = random.choice(regions)
        
        # Calculate base insurance cost with realistic factors
        base_cost = 1000
        
        # Age factor (costs increase with age)
        age_factor = (age - 18) * 50 + (age - 18) ** 1.2 * 10
        
        # BMI factor (higher costs for obesity)
        if bmi > 30:
            bmi_factor = (bmi - 30) * 200
        elif bmi < 18.5:
            bmi_factor = (18.5 - bmi) * 150
        else:
            bmi_factor = 0
        
        # Smoker factor (significantly higher costs)
        smoker_factor = 15000 if smoker == 'yes' else 0
        
        # Children factor (small increase per child)
        children_factor = children * 300
        
        # Region factor (different average costs by region)
        region_factors = {
            'northeast': 1.2,
            'northwest': 1.0,
            'southeast': 0.9,
            'southwest': 1.1
        }
        region_factor = region_factors[region]
        
        # Calculate total cost
        total_cost = (base_cost + age_factor + bmi_factor + smoker_factor + children_factor) * region_factor
        
        # Add some random noise
        noise = np.random.normal(0, total_cost * 0.1)
        total_cost = max(1000, total_cost + noise)
        
        # Round to 2 decimal places
        total_cost = round(total_cost, 2)
        
        data.append({
            'age': age,
            'sex': sex,
            'bmi': bmi,
            'children': children,
            'smoker': smoker,
            'region': region,
            'charges': total_cost
        })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Save to CSV
    df.to_csv('medical_insurance_dataset.csv', index=False)
    
    print(f"Dataset generated successfully!")
    print(f"Shape: {df.shape}")
    print(f"\nDataset Preview:")
    print(df.head(10))
    print(f"\nDataset Info:")
    print(df.describe())
    
    return df

if __name__ == "__main__":
    generate_medical_insurance_dataset(2000)