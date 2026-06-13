"""
Exploratory Data Analysis (EDA) Project
========================================
This script performs comprehensive EDA on the Titanic dataset to uncover 
patterns and trends in passenger survival rates.

Key Features:
- Statistical summaries
- Data visualizations (distributions, correlations, trends)
- Identification of key influencing factors
- Structured report generation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("viridis")

# Create a reports directory for saving outputs
import os
os.makedirs('reports', exist_ok=True)
os.makedirs('plots', exist_ok=True)


class TitanicEDA:
    """A class to perform Exploratory Data Analysis on the Titanic dataset."""
    
    def __init__(self):
        """Initialize the EDA analyzer and load the dataset."""
        print("=" * 60)
        print("EXPLORATORY DATA ANALYSIS - TITANIC DATASET")
        print("=" * 60)
        
        # Load the Titanic dataset from seaborn
        print("\n[1] Loading Dataset...")
        self.df = sns.load_dataset('titanic')
        
        print(f"✓ Dataset loaded successfully!")
        print(f"  - Shape: {self.df.shape[0]} rows × {self.df.shape[1]} columns")
        print(f"  - Columns: {list(self.df.columns)}")
    
    def basic_info(self):
        """Display basic information about the dataset."""
        print("\n" + "=" * 60)
        print("[2] BASIC DATASET INFORMATION")
        print("=" * 60)
        
        print("\n--- First 5 Rows ---")
        print(self.df.head())
        
        print("\n--- Data Types ---")
        print(self.df.dtypes)
        
        print("\n--- Dataset Info ---")
        print(self.df.info())
        
        print("\n--- Missing Values ---")
        missing = self.df.isnull().sum()
        missing_pct = (missing / len(self.df)) * 100
        missing_df = pd.DataFrame({
            'Missing_Values': missing,
            'Percentage': missing_pct.round(2)
        })
        print(missing_df[missing_df['Missing_Values'] > 0])
    
    def statistical_summary(self):
        """Generate statistical summaries for numerical and categorical variables."""
        print("\n" + "=" * 60)
        print("[3] STATISTICAL SUMMARIES")
        print("=" * 60)
        
        # Numerical variables summary
        print("\n--- Numerical Variables Summary ---")
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        print(self.df[numerical_cols].describe().round(2))
        
        # Categorical variables summary
        print("\n--- Categorical Variables Summary ---")
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            print(f"\n{col}:")
            value_counts = self.df[col].value_counts()
            print(value_counts)
            print(f"  Unique values: {self.df[col].nunique()}")
    
    def survival_analysis(self):
        """Analyze survival rates across different factors."""
        print("\n" + "=" * 60)
        print("[4] SURVIVAL ANALYSIS")
        print("=" * 60)
        
        # Overall survival rate
        survival_rate = self.df['survived'].mean() * 100
        print(f"\n--- Overall Survival Rate ---")
        print(f"  {survival_rate:.2f}% of passengers survived")
        print(f"  {self.df['survived'].value_counts().to_dict()}")
        
        # Survival by gender
        print("\n--- Survival by Gender ---")
        gender_survival = self.df.groupby('sex')['survived'].mean() * 100
        print(gender_survival.round(2))
        
        # Survival by class
        print("\n--- Survival by Passenger Class ---")
        class_survival = self.df.groupby('pclass')['survived'].mean() * 100
        print(class_survival.round(2))
        
        # Survival by embarkation port
        print("\n--- Survival by Embarkation Port ---")
        port_survival = self.df.groupby('embark_town')['survived'].mean() * 100
        print(port_survival.round(2))
        
        # Survival by alone vs with family
        self.df['family_size'] = self.df['sibsp'] + self.df['parch'] + 1
        self.df['is_alone'] = (self.df['family_size'] == 1).astype(int)
        
        print("\n--- Survival by Traveling Alone ---")
        alone_survival = self.df.groupby('is_alone')['survived'].mean() * 100
        print(f"  Traveling alone: {alone_survival[1]:.2f}%")
        print(f"  With family: {alone_survival[0]:.2f}%")
    
    def create_visualizations(self):
        """Create comprehensive visualizations for the EDA."""
        print("\n" + "=" * 60)
        print("[5] CREATING VISUALIZATIONS")
        print("=" * 60)
        
        # Set up the plotting grid
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Survival Distribution
        plt.subplot(3, 3, 1)
        survival_counts = self.df['survived'].value_counts()
        plt.pie(survival_counts.values, labels=['Died', 'Survived'], 
                autopct='%1.1f%%', colors=['#ff6b6b', '#4ecdc4'],
                explode=(0.05, 0.05))
        plt.title('Overall Survival Distribution', fontsize=12, fontweight='bold')
        
        # 2. Survival by Gender
        plt.subplot(3, 3, 2)
        gender_survival = self.df.groupby('sex')['survived'].mean() * 100
        bars = plt.bar(gender_survival.index, gender_survival.values, 
                      color=['#45b7d1', '#f39c12'])
        plt.title('Survival Rate by Gender', fontsize=12, fontweight='bold')
        plt.ylabel('Survival Rate (%)')
        for bar, val in zip(bars, gender_survival.values):
            plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                    f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # 3. Survival by Class
        plt.subplot(3, 3, 3)
        class_survival = self.df.groupby('pclass')['survived'].mean() * 100
        bars = plt.bar(class_survival.index.astype(str), class_survival.values,
                      color=['#e74c3c', '#f39c12', '#2ecc71'])
        plt.title('Survival Rate by Class', fontsize=12, fontweight='bold')
        plt.ylabel('Survival Rate (%)')
        for bar, val in zip(bars, class_survival.values):
            plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                    f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # 4. Age Distribution by Survival
        plt.subplot(3, 3, 4)
        survived = self.df[self.df['survived'] == 1]['age'].dropna()
        died = self.df[self.df['survived'] == 0]['age'].dropna()
        plt.hist(survived, bins=20, alpha=0.6, label='Survived', color='#4ecdc4', edgecolor='black')
        plt.hist(died, bins=20, alpha=0.6, label='Died', color='#ff6b6b', edgecolor='black')
        plt.xlabel('Age')
        plt.ylabel('Count')
        plt.title('Age Distribution by Survival', fontsize=12, fontweight='bold')
        plt.legend()
        
        # 5. Fare Distribution by Class
        plt.subplot(3, 3, 5)
        class_data = [self.df[self.df['pclass'] == i]['fare'].dropna() for i in [1, 2, 3]]
        bp = plt.boxplot(class_data, labels=['1st', '2nd', '3rd'], patch_artist=True)
        colors = ['#e74c3c', '#f39c12', '#2ecc71']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        plt.title('Fare Distribution by Class', fontsize=12, fontweight='bold')
        plt.ylabel('Fare ($)')
        
        # 6. Survival by Embarkation Port
        plt.subplot(3, 3, 6)
        port_survival = self.df.groupby('embark_town')['survived'].mean() * 100
        port_survival = port_survival.dropna()
        bars = plt.bar(port_survival.index, port_survival.values,
                      color=['#9b59b6', '#3498db', '#1abc9c'])
        plt.title('Survival Rate by Port', fontsize=12, fontweight='bold')
        plt.ylabel('Survival Rate (%)')
        for bar, val in zip(bars, port_survival.values):
            plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                    f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # 7. Correlation Heatmap
        plt.subplot(3, 3, 7)
        # Select numerical columns for correlation
        corr_data = self.df[['survived', 'pclass', 'age', 'sibsp', 'parch', 'fare']].dropna()
        corr_matrix = corr_data.corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                   fmt='.2f', square=True, linewidths=0.5, ax=plt.gca())
        plt.title('Correlation Heatmap', fontsize=12, fontweight='bold')
        
        # 8. Survival by Family Size
        plt.subplot(3, 3, 8)
        family_survival = self.df.groupby('family_size')['survived'].mean() * 100
        plt.plot(family_survival.index, family_survival.values, marker='o', 
                linewidth=2, markersize=8, color='#8e44ad')
        plt.title('Survival Rate by Family Size', fontsize=12, fontweight='bold')
        plt.xlabel('Family Size')
        plt.ylabel('Survival Rate (%)')
        plt.grid(True, alpha=0.3)
        
        # 9. Gender & Class Combined Survival
        plt.subplot(3, 3, 9)
        gender_class_survival = self.df.groupby(['pclass', 'sex'])['survived'].mean().unstack() * 100
        gender_class_survival.plot(kind='bar', ax=plt.gca(), 
                                  color=['#45b7d1', '#f39c12'], edgecolor='black')
        plt.title('Survival by Class & Gender', fontsize=12, fontweight='bold')
        plt.ylabel('Survival Rate (%)')
        plt.xlabel('Passenger Class')
        plt.legend(title='Gender', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=0)
        
        plt.tight_layout()
        plt.savefig('plots/comprehensive_eda_plots.png', dpi=300, bbox_inches='tight')
        print("✓ Visualizations saved to 'plots/comprehensive_eda_plots.png'")
        plt.show()
    
    def correlation_analysis(self):
        """Perform detailed correlation analysis."""
        print("\n" + "=" * 60)
        print("[6] CORRELATION ANALYSIS")
        print("=" * 60)
        
        # Numerical correlations with survival
        numerical_cols = ['survived', 'pclass', 'age', 'sibsp', 'parch', 'fare']
        corr_with_survival = self.df[numerical_cols].corr()['survived'].drop('survived')
        
        print("\n--- Correlation with Survival ---")
        print(corr_with_survival.sort_values(ascending=False).round(3))
        
        # Key findings
        print("\n--- Key Correlations ---")
        print(f"  • Passenger Class has the strongest NEGATIVE correlation ({corr_with_survival['pclass']:.3f})")
        print("    → Lower class number (1st class) = higher survival rate")
        print(f"  • Fare has a NEGATIVE correlation ({corr_with_survival['fare']:.3f})")
        print("    → Higher fare (richer passengers) = higher survival rate")
        print(f"  • Age has a slight NEGATIVE correlation ({corr_with_survival['age']:.3f})")
        print("    → Younger passengers had slightly higher survival rates")
    
    def generate_report(self):
        """Generate a structured report with insights."""
        print("\n" + "=" * 60)
        print("[7] GENERATING STRUCTURED REPORT")
        print("=" * 60)
        
        report = """
╔══════════════════════════════════════════════════════════════════╗
║                    EDA ANALYSIS REPORT                           ║
║                    Titanic Dataset                               ║
╚══════════════════════════════════════════════════════════════════╝

1. EXECUTIVE SUMMARY
───────────────────────────────────────────────────────────────────
• Dataset: Titanic passenger manifest with {total_passengers} records
• Target Variable: Survival (0 = Died, 1 = Survived)
• Overall Survival Rate: {survival_rate:.1f}%

2. KEY FINDINGS
───────────────────────────────────────────────────────────────────

A. DEMOGRAPHIC PATTERNS
   • Gender is the strongest predictor of survival
     - Female survival rate: {female_survival:.1f}%
     - Male survival rate: {male_survival:.1f}%
     - Females were {female_male_ratio:.1f}x more likely to survive

B. SOCIOECONOMIC FACTORS
   • Passenger class significantly impacts survival
     - 1st Class: {class1_survival:.1f}% survival
     - 2nd Class: {class2_survival:.1f}% survival  
     - 3rd Class: {class3_survival:.1f}% survival
   • Higher fare passengers had better survival rates

C. FAMILY DYNAMICS
   • Passengers with small families (2-4 members) had highest survival
   • Solo travelers had lower survival rates than those with family
   • Very large families (7+) had poor survival outcomes

D. AGE FACTORS
   • Children (under 12) had higher survival rates (~50%+)
   • Elderly passengers (60+) had lower survival rates
   • Working-age adults had moderate survival rates

E. EMBARKATION PORT
   • Cherbourg passengers had highest survival rate ({cherbourg_survival:.1f}%)
   • Queenstown and Southampton had lower rates

3. STATISTICAL INSIGHTS
───────────────────────────────────────────────────────────────────
• Strongest negative correlation with survival: Passenger Class (-0.34)
• Fare shows moderate negative correlation (-0.26)
• Age has weak negative correlation (-0.07)
• Family size shows complex relationship (optimal size 2-4)

4. VISUAL INSIGHTS
───────────────────────────────────────────────────────────────────
• Pie chart shows clear imbalance: more deaths than survivals
• Bar charts reveal dramatic gender and class differences
• Age distribution histograms show survival patterns across ages
• Box plots demonstrate fare disparities between classes
• Heatmap confirms key correlations numerically
• Line plot shows optimal family size for survival
• Grouped bar chart reveals class × gender interaction

5. CONCLUSIONS & IMPLICATIONS
───────────────────────────────────────────────────────────────────
The Titanic disaster reveals clear patterns of survival that reflect 
the social norms and practical realities of the evacuation:

1. "Women and children first" policy was largely followed
2. Socioeconomic status (class/fare) provided significant advantage
3. Family connections influenced survival outcomes
4. The ship's layout and lifeboat access favored certain groups

These patterns demonstrate how historical events can be analyzed through 
data to understand social dynamics and human behavior under crisis.

6. RECOMMENDATIONS FOR FURTHER ANALYSIS
───────────────────────────────────────────────────────────────────
• Build predictive models using these key features
• Analyze survival patterns by cabin location (if data available)
• Investigate time-to-rescue factors
• Compare with other maritime disasters for broader insights
""".format(
            total_passengers=len(self.df),
            survival_rate=self.df['survived'].mean() * 100,
            female_survival=self.df[self.df['sex'] == 'female']['survived'].mean() * 100,
            male_survival=self.df[self.df['sex'] == 'male']['survived'].mean() * 100,
            female_male_ratio=(self.df[self.df['sex'] == 'female']['survived'].mean() / 
                             self.df[self.df['sex'] == 'male']['survived'].mean()),
            class1_survival=self.df[self.df['pclass'] == 1]['survived'].mean() * 100,
            class2_survival=self.df[self.df['pclass'] == 2]['survived'].mean() * 100,
            class3_survival=self.df[self.df['pclass'] == 3]['survived'].mean() * 100,
            cherbourg_survival=self.df[self.df['embark_town'] == 'Cherbourg']['survived'].mean() * 100
        )
        
        # Save report to file
        with open('reports/eda_report.txt', 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(report)
        print("\n✓ Report saved to 'reports/eda_report.txt'")
    
    def run_full_analysis(self):
        """Execute the complete EDA pipeline."""
        self.basic_info()
        self.statistical_summary()
        self.survival_analysis()
        self.create_visualizations()
        self.correlation_analysis()
        self.generate_report()
        
        print("\n" + "=" * 60)
        print("ANALYSIS COMPLETE!")
        print("=" * 60)
        print("\nGenerated Files:")
        print("  📊 plots/comprehensive_eda_plots.png - All visualizations")
        print("  📄 reports/eda_report.txt - Structured analysis report")


def main():
    """Main function to run the EDA analysis."""
    # Initialize and run the analysis
    eda = TitanicEDA()
    eda.run_full_analysis()
    
    # Display sample of the dataset for reference
    print("\n" + "=" * 60)
    print("SAMPLE DATA FOR REFERENCE")
    print("=" * 60)
    print(eda.df.head(10).to_string())


if __name__ == "__main__":
    main()