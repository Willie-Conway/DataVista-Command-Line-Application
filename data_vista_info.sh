```bash
#!/bin/bash

# ============================================================
# DataVista CLI Application - Information & Help System
# Version: 2.0.0
# Author: Willie Conway
# Last Updated: January 28, 2025
# ============================================================

VERSION="2.0.0"
CONFIG_FILE="${HOME}/.datavista/config.yaml"
LOG_FILE="${HOME}/.datavista/datavista.log"

# Color Codes for better output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
WHITE='\033[1;37m'
BOLD='\033[1m'
UNDERLINE='\033[4m'
NC='\033[0m' # No Color

# ASCII Art Banner
print_banner() {
    echo -e "${BLUE}"
    cat << "EOF"
    ██████╗  █████╗ ████████╗ █████╗ ██╗   ██╗██╗███████╗████████╗ █████╗ 
    ██╔══██╗██╔══██╗╚══██╔══╝██╔══██╗██║   ██║██║██╔════╝╚══██╔══╝██╔══██╗
    ██║  ██║███████║   ██║   ███████║██║   ██║██║███████╗   ██║   ███████║
    ██║  ██║██╔══██║   ██║   ██╔══██║╚██╗ ██╔╝██║╚════██║   ██║   ██╔══██║
    ██████╔╝██║  ██║   ██║   ██║  ██║ ╚████╔╝ ██║███████║   ██║   ██║  ██║
    ╚═════╝ ╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝  ╚═══╝  ╚═╝╚══════╝   ╚═╝   ╚═╝  ╚═╝
EOF
    echo -e "${NC}"
    echo -e "${GREEN}=================== Data Science CLI Platform ===================${NC}"
    echo -e "${YELLOW}Version: ${VERSION} | Author: Willie Conway | License: MIT${NC}"
    echo ""
}

# Check if configuration directory exists
check_config() {
    if [ ! -d "${HOME}/.datavista" ]; then
        echo -e "${YELLOW}Creating DataVista configuration directory...${NC}"
        mkdir -p "${HOME}/.datavista"
        
        # Create default configuration
        cat > "${CONFIG_FILE}" << EOF
# DataVista Configuration File
# Created: $(date)

data_vista:
  general:
    version: "${VERSION}"
    log_level: "INFO"
    cache_dir: "${HOME}/.datavista/cache"
    models_dir: "${HOME}/.datavista/models"
    reports_dir: "${HOME}/.datavista/reports"
  
  logging:
    file: "${LOG_FILE}"
    max_size_mb: 10
    backup_count: 5
    format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  
  visualization:
    style: "seaborn"
    palette: "viridis"
    dpi: 150
    default_format: "png"
    interactive_backend: "plotly"
  
  machine_learning:
    cross_validation_folds: 5
    random_state: 42
    test_size: 0.2
    scoring_metric: "accuracy"
    hyperparameter_tuning: true
  
  performance:
    max_memory_gb: 8
    chunk_size: 10000
    parallel_processing: true
    n_jobs: -1
  
  directories:
    data: "./data"
    models: "./models"
    results: "./results"
    exports: "./exports"
    templates: "./templates"
EOF
        
        # Create subdirectories
        mkdir -p "${HOME}/.datavista/cache"
        mkdir -p "${HOME}/.datavista/models"
        mkdir -p "${HOME}/.datavista/reports"
        
        echo -e "${GREEN}✓ Configuration created: ${CONFIG_FILE}${NC}"
    fi
}

# Display main menu
show_main_menu() {
    echo -e "\n${CYAN}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║                  ${WHITE}DATA VISTA MAIN MENU${CYAN}                    ║${NC}"
    echo -e "${CYAN}╠════════════════════════════════════════════════════════════╣${NC}"
    echo -e "${CYAN}║                                                            ║${NC}"
    echo -e "${CYAN}║  ${GREEN}1.${NC} ${WHITE}📊  Statistical Analysis${NC}                        ${CYAN}║${NC}"
    echo -e "${CYAN}║  ${GREEN}2.${NC} ${WHITE}🤖  Machine Learning${NC}                            ${CYAN}║${NC}"
    echo -e "${CYAN}║  ${GREEN}3.${NC} ${WHITE}📈  Data Visualization${NC}                          ${CYAN}║${NC}"
    echo -e "${CYAN}║  ${GREEN}4.${NC} ${WHITE}🔬  Hypothesis Testing${NC}                          ${CYAN}║${NC}"
    echo -e "${CYAN}║  ${GREEN}5.${NC} ${WHITE}🎯  Clustering Analysis${NC}                         ${CYAN}║${NC}"
    echo -e "${CYAN}║  ${GREEN}6.${NC} ${WHITE}📅  Time Series Forecasting${NC}                     ${CYAN}║${NC}"
    echo -e "${CYAN}║  ${GREEN}7.${NC} ${WHITE}💾  Model Management${NC}                           ${CYAN}║${NC}"
    echo -e "${CYAN}║  ${GREEN}8.${NC} ${WHITE}⚙️   Settings & Configuration${NC}                   ${CYAN}║${NC}"
    echo -e "${CYAN}║  ${GREEN}9.${NC} ${WHITE}📚  Documentation & Examples${NC}                    ${CYAN}║${NC}"
    echo -e "${CYAN}║  ${GREEN}10.${NC} ${WHITE}🚀  Quick Start Guide${NC}                          ${CYAN}║${NC}"
    echo -e "${CYAN}║  ${GREEN}11.${NC} ${WHITE}❓  Help & Troubleshooting${NC}                     ${CYAN}║${NC}"
    echo -e "${CYAN}║  ${GREEN}12.${NC} ${WHITE}🚪  Exit${NC}                                      ${CYAN}║${NC}"
    echo -e "${CYAN}║                                                            ║${NC}"
    echo -e "${CYAN}╚════════════════════════════════════════════════════════════╝${NC}"
    echo ""
}

# Display module submenus
show_statistical_menu() {
    echo -e "\n${YELLOW}📊 STATISTICAL ANALYSIS MODULE${NC}"
    echo -e "${GREEN}═══════════════════════════════════════════════════${NC}"
    echo "1.  Descriptive Statistics"
    echo "2.  Correlation Analysis"
    echo "3.  Distribution Analysis"
    echo "4.  Outlier Detection"
    echo "5.  Normality Testing"
    echo "6.  Advanced Statistical Tests"
    echo "7.  Data Profiling Report"
    echo "8.  Statistical Summary Export"
    echo "9.  Back to Main Menu"
    echo ""
}

show_ml_menu() {
    echo -e "\n${YELLOW}🤖 MACHINE LEARNING MODULE${NC}"
    echo -e "${GREEN}═══════════════════════════════════════════════════${NC}"
    echo "1.  Supervised Learning"
    echo "    └─ Regression (Linear, Random Forest, XGBoost)"
    echo "    └─ Classification (Logistic, SVM, Ensemble)"
    echo "2.  Unsupervised Learning"
    echo "    └─ Clustering (K-Means, DBSCAN, Hierarchical)"
    echo "    └─ Dimensionality Reduction (PCA, t-SNE)"
    echo "3.  AutoML (Automatic Algorithm Selection)"
    echo "4.  Model Evaluation & Validation"
    echo "5.  Feature Importance Analysis"
    echo "6.  Hyperparameter Tuning"
    echo "7.  Ensemble Methods"
    echo "8.  Model Comparison"
    echo "9.  Back to Main Menu"
    echo ""
}

show_visualization_menu() {
    echo -e "\n${YELLOW}📈 DATA VISUALIZATION MODULE${NC}"
    echo -e "${GREEN}═══════════════════════════════════════════════════${NC}"
    echo "1.  Distribution Plots"
    echo "    └─ Histograms, KDE, Box Plots, Violin Plots"
    echo "2.  Relationship Plots"
    echo "    └─ Scatter, Pair, Correlation Heatmaps"
    echo "3.  Categorical Plots"
    echo "    └─ Bar, Count, Pie, Donut Charts"
    echo "4.  Time Series Plots"
    echo "    └─ Line, Area, Seasonal Decomposition"
    echo "5.  Statistical Plots"
    echo "    └─ QQ, Probability, Residual Plots"
    echo "6.  Geographical Plots (if coordinates available)"
    echo "7.  Interactive Plots (Plotly)"
    echo "8.  Dashboard Creation"
    echo "9.  Back to Main Menu"
    echo ""
}

# Display quick start guide
show_quick_start() {
    echo -e "\n${YELLOW}🚀 QUICK START GUIDE${NC}"
    echo -e "${GREEN}═══════════════════════════════════════════════════${NC}"
    
    echo -e "\n${WHITE}1. Installation:${NC}"
    echo "   git clone https://github.com/Willie-Conway/DataVista-Command-Line-Application.git"
    echo "   cd DataVista-Command-Line-Application"
    echo "   pip install -r requirements.txt"
    
    echo -e "\n${WHITE}2. Basic Usage:${NC}"
    echo "   # Run with sample data"
    echo "   python src/data_vista.py"
    
    echo "   # Run with custom data"
    echo "   python src/data_vista.py --data data/your_dataset.csv"
    
    echo -e "\n${WHITE}3. Common Commands:${NC}"
    echo "   --data <file>        Load specific dataset"
    echo "   --analysis <type>    Type of analysis (stats, ml, viz)"
    echo "   --target <column>    Target column for ML"
    echo "   --output <dir>       Output directory"
    echo "   --config <file>      Custom configuration"
    
    echo -e "\n${WHITE}4. Example Workflows:${NC}"
    echo "   # Complete analysis pipeline"
    echo "   python src/data_vista.py --data data/sales.csv --analysis complete"
    
    echo "   # Only statistical analysis"
    echo "   python src/data_vista.py --data data/sales.csv --analysis stats"
    
    echo "   # ML with specific algorithm"
    echo "   python src/data_vista.py --data data/churn.csv --target churn --algorithm random_forest"
    
    echo -e "\n${WHITE}5. Advanced Features:${NC}"
    echo "   • Parallel processing for large datasets"
    echo "   • Automated feature engineering"
    echo "   • Model versioning and tracking"
    echo "   • Interactive HTML reports"
    echo "   • Docker containerization"
}

# Display documentation
show_documentation() {
    echo -e "\n${YELLOW}📚 DOCUMENTATION & RESOURCES${NC}"
    echo -e "${GREEN}═══════════════════════════════════════════════════${NC}"
    
    echo -e "\n${WHITE}Available Documentation:${NC}"
    echo "1.  User Guide              - Complete usage instructions"
    echo "2.  API Reference           - Module and function documentation"
    echo "3.  Examples Directory      - Sample scripts and workflows"
    echo "4.  Tutorial Videos         - Step-by-step video guides"
    echo "5.  Troubleshooting Guide   - Common issues and solutions"
    echo "6.  Development Guide       - Contributing and extending DataVista"
    echo "7.  Best Practices          - Recommended workflows"
    echo "8.  Performance Tips        - Optimizing for large datasets"
    
    echo -e "\n${WHITE}Online Resources:${NC}"
    echo "• GitHub Repository: https://github.com/Willie-Conway/DataVista-Command-Line-Application"
    echo "• Issue Tracker:     https://github.com/Willie-Conway/DataVista-Command-Line-Application/issues"
    echo "• Documentation:     https://github.com/Willie-Conway/DataVista-Command-Line-Application/tree/main/docs"
    echo "• Wiki:              https://github.com/Willie-Conway/DataVista-Command-Line-Application/wiki"
    
    echo -e "\n${WHITE}Sample Datasets:${NC}"
    echo "• data/sample_data.csv             - General purpose sample"
    echo "• data/customer_churn.csv          - Customer analytics"
    echo "• data/market_research.csv         - Market analysis"
    echo "• data/walmart_grocery_data.csv    - Retail analytics"
    echo "• data/Age_Income_Dataset.csv      - Demographic analysis"
}

# Display help and troubleshooting
show_help() {
    echo -e "\n${YELLOW}❓ HELP & TROUBLESHOOTING${NC}"
    echo -e "${GREEN}═══════════════════════════════════════════════════${NC}"
    
    echo -e "\n${WHITE}Common Issues:${NC}"
    
    echo -e "\n${CYAN}Issue 1: Missing Dependencies${NC}"
    echo "Symptoms: ImportError or ModuleNotFoundError"
    echo "Solution: Run 'pip install -r requirements.txt'"
    echo "          Or 'pip install pandas numpy scikit-learn matplotlib seaborn'"
    
    echo -e "\n${CYAN}Issue 2: File Not Found${NC}"
    echo "Symptoms: FileNotFoundError"
    echo "Solution: Check file path and permissions"
    echo "          Use absolute paths: --data /full/path/to/file.csv"
    
    echo -e "\n${CYAN}Issue 3: Memory Issues${NC}"
    echo "Symptoms: MemoryError or slow performance"
    echo "Solution: Use chunk processing: --chunk-size 10000"
    echo "          Reduce features: --feature-selection"
    echo "          Use sampling: --sample 0.1"
    
    echo -e "\n${CYAN}Issue 4: Plotting Errors${NC}"
    echo "Symptoms: Matplotlib or Seaborn errors"
    echo "Solution: Set backend: --backend agg"
    echo "          Install GUI backend: pip install PyQt5"
    
    echo -e "\n${WHITE}Debugging Tips:${NC}"
    echo "1. Enable verbose logging: --verbose"
    echo "2. Check log file: cat ~/.datavista/datavista.log"
    echo "3. Run in debug mode: --debug"
    echo "4. Test with sample data first"
    
    echo -e "\n${WHITE}Getting Support:${NC}"
    echo "1. Check existing issues on GitHub"
    echo "2. Create new issue with:"
    echo "   - Error message"
    echo "   - DataVista version"
    echo "   - Python version"
    echo "   - Steps to reproduce"
    echo "3. Join community discussions"
}

# Display system information
show_system_info() {
    echo -e "\n${YELLOW}⚙️  SYSTEM INFORMATION${NC}"
    echo -e "${GREEN}═══════════════════════════════════════════════════${NC}"
    
    echo -e "\n${WHITE}DataVista Configuration:${NC}"
    echo "Version:          ${VERSION}"
    echo "Config File:      ${CONFIG_FILE}"
    echo "Log File:         ${LOG_FILE}"
    
    echo -e "\n${WHITE}System Information:${NC}"
    echo "Python Version:   $(python --version 2>/dev/null || echo 'Not found')"
    echo "Pandas Version:   $(python -c "import pandas; print(pandas.__version__)" 2>/dev/null || echo 'Not installed')"
    echo "NumPy Version:    $(python -c "import numpy; print(numpy.__version__)" 2>/dev/null || echo 'Not installed')"
    echo "Scikit-Learn:     $(python -c "import sklearn; print(sklearn.__version__)" 2>/dev/null || echo 'Not installed')"
    
    echo -e "\n${WHITE}Disk Usage:${NC}"
    if [ -d "${HOME}/.datavista" ]; then
        du -sh "${HOME}/.datavista"
    else
        echo "Config directory not found"
    fi
}

# Display settings menu
show_settings_menu() {
    echo -e "\n${YELLOW}⚙️  SETTINGS & CONFIGURATION${NC}"
    echo -e "${GREEN}═══════════════════════════════════════════════════${NC}"
    
    echo "1.  View Current Configuration"
    echo "2.  Edit Configuration File"
    echo "3.  Reset to Defaults"
    echo "4.  Change Logging Level"
    echo "5.  Configure Directories"
    echo "6.  Set Default Parameters"
    echo "7.  Export Configuration"
    echo "8.  Import Configuration"
    echo "9.  Test Configuration"
    echo "10. Back to Main Menu"
    echo ""
}

# Run command based on selection
run_command() {
    local choice="$1"
    
    case $choice in
        1) # Statistical Analysis
            show_statistical_menu
            read -p "Select statistical analysis option: " stats_choice
            handle_statistical_choice "$stats_choice"
            ;;
            
        2) # Machine Learning
            show_ml_menu
            read -p "Select ML option: " ml_choice
            handle_ml_choice "$ml_choice"
            ;;
            
        3) # Data Visualization
            show_visualization_menu
            read -p "Select visualization option: " viz_choice
            handle_viz_choice "$viz_choice"
            ;;
            
        4) # Hypothesis Testing
            echo -e "\n${GREEN}Running Hypothesis Testing Module...${NC}"
            python src/data_vista.py --analysis hypothesis
            ;;
            
        5) # Clustering Analysis
            echo -e "\n${GREEN}Running Clustering Analysis Module...${NC}"
            python src/data_vista.py --analysis clustering
            ;;
            
        6) # Time Series Forecasting
            echo -e "\n${GREEN}Running Time Series Forecasting Module...${NC}"
            python src/data_vista.py --analysis timeseries
            ;;
            
        7) # Model Management
            echo -e "\n${GREEN}Opening Model Management Interface...${NC}"
            python src/data_vista.py --model-manage
            ;;
            
        8) # Settings
            show_settings_menu
            read -p "Select settings option: " settings_choice
            handle_settings_choice "$settings_choice"
            ;;
            
        9) # Documentation
            show_documentation
            ;;
            
        10) # Quick Start Guide
            show_quick_start
            ;;
            
        11) # Help & Troubleshooting
            show_help
            ;;
            
        12) # Exit
            echo -e "\n${GREEN}Thank you for using DataVista!${NC}"
            exit 0
            ;;
            
        *)
            echo -e "\n${RED}Invalid option. Please try again.${NC}"
            ;;
    esac
}

# Handle statistical analysis choices
handle_statistical_choice() {
    local choice="$1"
    
    case $choice in
        1)
            echo -e "\n${GREEN}Running Descriptive Statistics...${NC}"
            python src/data_vista.py --analysis descriptive
            ;;
        2)
            echo -e "\n${GREEN}Running Correlation Analysis...${NC}"
            python src/data_vista.py --analysis correlation
            ;;
        3)
            echo -e "\n${GREEN}Running Distribution Analysis...${NC}"
            python src/data_vista.py --analysis distribution
            ;;
        4)
            echo -e "\n${GREEN}Running Outlier Detection...${NC}"
            python src/data_vista.py --analysis outliers
            ;;
        5)
            echo -e "\n${GREEN}Running Normality Testing...${NC}"
            python src/data_vista.py --analysis normality
            ;;
        6)
            echo -e "\n${GREEN}Running Advanced Statistical Tests...${NC}"
            python src/data_vista.py --analysis advanced-stats
            ;;
        7)
            echo -e "\n${GREEN}Generating Data Profiling Report...${NC}"
            python src/data_vista.py --analysis profile
            ;;
        8)
            echo -e "\n${GREEN}Exporting Statistical Summary...${NC}"
            python src/data_vista.py --analysis export-stats
            ;;
        9)
            return
            ;;
        *)
            echo -e "${RED}Invalid option${NC}"
            ;;
    esac
}

# Handle machine learning choices
handle_ml_choice() {
    local choice="$1"
    
    case $choice in
        1)
            echo -e "\n${GREEN}Running Supervised Learning...${NC}"
            read -p "Enter target column: " target
            read -p "Enter algorithm (linear, random_forest, xgboost): " algorithm
            python src/data_vista.py --analysis ml --target "$target" --algorithm "$algorithm"
            ;;
        2)
            echo -e "\n${GREEN}Running Unsupervised Learning...${NC}"
            read -p "Enter clustering algorithm (kmeans, dbscan, hierarchical): " algorithm
            python src/data_vista.py --analysis clustering --algorithm "$algorithm"
            ;;
        3)
            echo -e "\n${GREEN}Running AutoML...${NC}"
            read -p "Enter target column: " target
            python src/data_vista.py --analysis automl --target "$target"
            ;;
        4)
            echo -e "\n${GREEN}Running Model Evaluation...${NC}"
            python src/data_vista.py --analysis evaluate
            ;;
        5)
            echo -e "\n${GREEN}Running Feature Importance Analysis...${NC}"
            python src/data_vista.py --analysis feature-importance
            ;;
        6)
            echo -e "\n${GREEN}Running Hyperparameter Tuning...${NC}"
            python src/data_vista.py --analysis tune
            ;;
        7)
            echo -e "\n${GREEN}Running Ensemble Methods...${NC}"
            python src/data_vista.py --analysis ensemble
            ;;
        8)
            echo -e "\n${GREEN}Running Model Comparison...${NC}"
            python src/data_vista.py --analysis compare-models
            ;;
        9)
            return
            ;;
        *)
            echo -e "${RED}Invalid option${NC}"
            ;;
    esac
}

# Handle visualization choices
handle_viz_choice() {
    local choice="$1"
    
    case $choice in
        1)
            echo -e "\n${GREEN}Creating Distribution Plots...${NC}"
            read -p "Enter column name(s): " columns
            python src/data_vista.py --visualize distribution --columns "$columns"
            ;;
        2)
            echo -e "\n${GREEN}Creating Relationship Plots...${NC}"
            read -p "Enter x column: " xcol
            read -p "Enter y column: " ycol
            python src/data_vista.py --visualize scatter --x "$xcol" --y "$ycol"
            ;;
        3)
            echo -e "\n${GREEN}Creating Categorical Plots...${NC}"
            read -p "Enter categorical column: " catcol
            python src/data_vista.py --visualize categorical --column "$catcol"
            ;;
        4)
            echo -e "\n${GREEN}Creating Time Series Plots...${NC}"
            read -p "Enter date column: " datecol
            read -p "Enter value column: " valuecol
            python src/data_vista.py --visualize timeseries --date "$datecol" --value "$valuecol"
            ;;
        5)
            echo -e "\n${GREEN}Creating Statistical Plots...${NC}"
            python src/data_vista.py --visualize statistical
            ;;
        6)
            echo -e "\n${GREEN}Creating Geographical Plots...${NC}"
            read -p "Enter latitude column: " lat
            read -p "Enter longitude column: " lon
            python src/data_vista.py --visualize geographical --lat "$lat" --lon "$lon"
            ;;
        7)
            echo -e "\n${GREEN}Creating Interactive Plots...${NC}"
            python src/data_vista.py --visualize interactive
            ;;
        8)
            echo -e "\n${GREEN}Creating Dashboard...${NC}"
            python src/data_vista.py --visualize dashboard
            ;;
        9)
            return
            ;;
        *)
            echo -e "${RED}Invalid option${NC}"
            ;;
    esac
}

# Handle settings choices
handle_settings_choice() {
    local choice="$1"
    
    case $choice in
        1)
            echo -e "\n${GREEN}Current Configuration:${NC}"
            if [ -f "$CONFIG_FILE" ]; then
                cat "$CONFIG_FILE"
            else
                echo "Configuration file not found"
            fi
            ;;
            
        2)
            if [ -f "$CONFIG_FILE" ]; then
                ${EDITOR:-vi} "$CONFIG_FILE"
                echo -e "${GREEN}Configuration updated${NC}"
            else
                echo -e "${RED}Configuration file not found${NC}"
            fi
            ;;
            
        3)
            read -p "Are you sure you want to reset to defaults? (y/n): " confirm
            if [[ $confirm == "y" || $confirm == "Y" ]]; then
                rm -rf "${HOME}/.datavista"
                check_config
                echo -e "${GREEN}Configuration reset to defaults${NC}"
            fi
            ;;
            
        4)
            echo -e "\n${GREEN}Current log level: $(grep -i "log_level" "$CONFIG_FILE" | cut -d: -f2 | tr -d ' "')${NC}"
            read -p "Enter new log level (DEBUG, INFO, WARNING, ERROR): " loglevel
            sed -i "s/log_level:.*/log_level: \"$loglevel\"/" "$CONFIG_FILE"
            echo -e "${GREEN}Log level updated to $loglevel${NC}"
            ;;
            
        5)
            echo -e "\n${GREEN}Current directories:${NC}"
            grep -A5 "directories:" "$CONFIG_FILE"
            echo ""
            read -p "Enter new data directory: " datadir
            sed -i "s|data:.*|data: \"$datadir\"|" "$CONFIG_FILE"
            echo -e "${GREEN}Data directory updated${NC}"
            ;;
            
        9)
            echo -e "\n${GREEN}Testing configuration...${NC}"
            if python -c "import yaml; yaml.safe_load(open('$CONFIG_FILE'))"; then
                echo -e "${GREEN}✓ Configuration is valid${NC}"
            else
                echo -e "${RED}✗ Configuration contains errors${NC}"
            fi
            ;;
            
        10)
            return
            ;;
            
        *)
            echo -e "${RED}Invalid option${NC}"
            ;;
    esac
}

# Main function
main() {
    # Clear screen
    clear
    
    # Print banner
    print_banner
    
    # Check and create configuration
    check_config
    
    # Show system information
    show_system_info
    
    # Main loop
    while true; do
        show_main_menu
        read -p "Select an option (1-12): " main_choice
        
        # Validate input
        if [[ ! "$main_choice" =~ ^[0-9]+$ ]] || [ "$main_choice" -lt 1 ] || [ "$main_choice" -gt 12 ]; then
            echo -e "${RED}Please enter a number between 1 and 12${NC}"
            sleep 1
            continue
        fi
        
        run_command "$main_choice"
        
        # Pause before showing menu again
        if [ "$main_choice" -ne 12 ]; then
            echo ""
            read -p "Press Enter to continue..."
        fi
    done
}

# Check if script is being sourced or executed
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    # Check for command line arguments
    if [ $# -eq 0 ]; then
        # No arguments, run interactive mode
        main
    else
        # Handle command line arguments
        case "$1" in
            --version|-v)
                echo "DataVista version $VERSION"
                ;;
            --help|-h)
                print_banner
                show_quick_start
                ;;
            --config)
                if [ -f "$CONFIG_FILE" ]; then
                    cat "$CONFIG_FILE"
                else
                    echo "Configuration file not found"
                fi
                ;;
            --info)
                print_banner
                show_system_info
                ;;
            --run)
                if [ $# -ge 2 ]; then
                    python src/data_vista.py "${@:2}"
                else
                    echo "Please specify arguments for data_vista.py"
                fi
                ;;
            *)
                echo "Usage: $0 [OPTION]"
                echo "Options:"
                echo "  --version, -v     Show version information"
                echo "  --help, -h        Show help information"
                echo "  --config          Show configuration"
                echo "  --info            Show system information"
                echo "  --run             Run DataVista with arguments"
                echo "  (no arguments)    Start interactive mode"
                ;;
        esac
    fi
fi

# ============================================================
# END OF SCRIPT
# ============================================================
```
