
# ollama models

## cloud
https://ollama.com/library/glm-4.7-flash
https://ollama.com/library/glm-4.7
https://ollama.com/library/neural-chat
## local
https://ollama.com/library/deepseek-ocr
https://ollama.com/library/lfm2.5-thinking
https://ollama.com/library/qwen2.5-coder
https://ollama.com/library/moondream
https://ollama.com/library/embeddinggemma
https://ollama.com/library/dolphin-phi
https://ollama.com/library/smollm2

# hardware constraint
mac air m1
ram 8gb ram


# llama index examples

{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "57ca0f53-f780-4cc5-8c37-d754c32a97cc",
   "metadata": {},
   "source": [
    "## Long-Form Document Extraction: Mining Information from SEC 10-K/Q Forms\n",
    "\n",
    "Companies listed on the US stock exchanges are required to file annual and quarterly reports with the SEC. These reports are called 10K (annual) and 10Q (quarterly) filings.\n",
    "10K/Q filings are information dense and contain a lot of information about the company's business, operations, and financials.\n",
    "The documents have a loosely defined structure and the reported metrics and sections may differ based on the company's operations. \n",
    "\n",
    "That said, there are enough commonalities that we may want to extract the information in a standardized format for downstream analysis. e.g. this could be \n",
    "used to extract financial metrics for a company and analysis of key risk factors after every earnings release.\n",
    "\n",
    "Let's take a look at Nvidia's 10-K filing for the year 2024. Here's the SEC link for the [10-K filing](https://www.sec.gov/ix?doc=/Archives/edgar/data/0001045810/000104581025000023/nvda-20250126.htm).\n",
    "As you can see, this is a pretty large document with a lot of information to parse through. \n",
    "\n",
    "> **Note:** This principle of what fields generalize across your target documents and what might be optional is an important one to keep in mind when designing your schema. \n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "355adfd4",
   "metadata": {},
   "source": [
    "> **⚠️ DEPRECATION NOTICE**>> This example uses the deprecated `llama-cloud-services` package, which will be maintained until **May 1, 2026**.>> **Please migrate to:**> - **Python**: `pip install llama-cloud>=1.0` ([GitHub](https://github.com/run-llama/llama-cloud-py))> - **New Package Documentation**: https://docs.cloud.llamaindex.ai/>> The new package provides the same functionality with improved performance and support."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "53c9d9e4",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "\n",
       "        <iframe\n",
       "            width=\"600\"\n",
       "            height=\"400\"\n",
       "            src=\"./data/sec_filings/nvda_10k.pdf\"\n",
       "            frameborder=\"0\"\n",
       "            allowfullscreen\n",
       "            \n",
       "        ></iframe>\n",
       "        "
      ],
      "text/plain": [
       "<IPython.lib.display.IFrame at 0x11b2e3850>"
      ]
     },
     "execution_count": null,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from IPython.display import IFrame\n",
    "\n",
    "IFrame(src=\"./data/sec_filings/nvda_10k.pdf\", width=600, height=400)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "3b2b0485",
   "metadata": {},
   "source": [
    "Let us initialize the LlamaExtract client to extract our information of interest from these 10-K filings. "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "55bfb70e",
   "metadata": {},
   "outputs": [],
   "source": [
    "from dotenv import load_dotenv\n",
    "from llama_cloud_services import LlamaExtract\n",
    "\n",
    "\n",
    "# Load environment variables (put LLAMA_CLOUD_API_KEY in your .env file)\n",
    "load_dotenv(override=True)\n",
    "\n",
    "# Optionally, add your project id/organization id\n",
    "llama_extract = LlamaExtract()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "f3c60767",
   "metadata": {},
   "source": [
    "### 1. Defining the Extraction Schema\n",
    "\n",
    "To begin with, we'll focus on extracting the following information from the 10K/Q filings which are common across different companies:\n",
    "- *Filing Information*: Date of filing, type of filing, reporting period end date, fiscal year, fiscal quarter\n",
    "- *Company Profile*: Name, ticker, reporting currency, stock exchanges, auditor\n",
    "- *Financial Highlights*: Key metrics to assess the company's financial health - revenue, gross profit, operating income, net income, EPS, EBITDA, free cash flow\n",
    "- *Business/Geographic Segments*: Revenue, operating income, year-over-year growth, outlook for each segment.\n",
    "- *Risk Factors*: Key risks as identified by the company management.\n",
    "- *Management Discussion & Analysis (MD&A)*: Key highlights from management discussion and analysis.\n",
    "\n",
    "\n",
    "#### Using Pydantic Models for Schema Definition\n",
    "\n",
    "We can use JSON to define the schema for the extraction or use Pydantic models to encapsulate the schema. In this example, we'll use Pydantic models for schema definition for a few reasons:\n",
    "- **Extensibility**: They are more flexible, easier to extend and maintain. \n",
    "- **Readability**: Pydantic models are more readable (less verbose) and easier to understand. Nested models in particular are easier to read than deeply nested JSON schemas.\n",
    "- **Type Safety**: By validating against the Pydantic model, your code is guaranteed to be type-safe for use downstream an part of an automated process. e.g. an extracted date field will not suddenly become a numeric type.\n",
    "\n",
    "In this case, imagine that you have a daily ETL pipeline that searches for new 10-K/Q filings and extracts the relevant information for these companies. Once the extraction results are available in LlamaExtract, *it is guaranteed to comply with the schema definition and can be sent to the ETL pipeline without worrying about data type mismatches.*\n",
    "\n",
    "We consider some key design considerations for the schema definition below."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "899569db",
   "metadata": {},
   "outputs": [],
   "source": [
    "from typing import Literal, Optional, List\n",
    "from pydantic import BaseModel, Field\n",
    "\n",
    "\n",
    "class FilingInfo(BaseModel):\n",
    "    \"\"\"Basic information about the SEC filing\"\"\"\n",
    "\n",
    "    filing_type: Literal[\"10-K\", \"10-Q\", \"10-K/A\", \"10-Q/A\"] = Field(\n",
    "        description=\"Type of SEC filing\"\n",
    "    )\n",
    "    filing_date: str = Field(description=\"Date when filing was submitted to SEC\")\n",
    "    reporting_period_end: str = Field(description=\"End date of reporting period\")\n",
    "    fiscal_year: int = Field(description=\"Fiscal year\")\n",
    "    fiscal_quarter: int = Field(description=\"Fiscal quarter (if 10-Q)\", ge=1, le=4)\n",
    "\n",
    "\n",
    "class CompanyProfile(BaseModel):\n",
    "    \"\"\"Essential company information\"\"\"\n",
    "\n",
    "    name: str = Field(description=\"Legal name of company\")\n",
    "    ticker: str = Field(description=\"Stock ticker symbol\")\n",
    "    reporting_currency: str = Field(description=\"Currency used in financial statements\")\n",
    "    exchanges: Optional[List[str]] = Field(\n",
    "        None, description=\"Stock exchanges where listed\"\n",
    "    )\n",
    "    auditor: Optional[str] = Field(None, description=\"Company's auditor\")\n",
    "\n",
    "\n",
    "class FinancialHighlights(BaseModel):\n",
    "    \"\"\"Key financial metrics from this reporting period\"\"\"\n",
    "\n",
    "    period_end: str = Field(description=\"End date of reporting period\")\n",
    "    comparison_period_end: Optional[str] = Field(\n",
    "        None, description=\"End date of comparison period (typically prior year/quarter)\"\n",
    "    )\n",
    "    currency: str = Field(description=\"Currency of financial figures\")\n",
    "    unit: str = Field(\n",
    "        description=\"Unit of financial figures (thousands, millions, etc.)\"\n",
    "    )\n",
    "    revenue: float = Field(description=\"Total revenue for period\")\n",
    "    revenue_prior_period: Optional[float] = Field(\n",
    "        None, description=\"Revenue from comparison period\"\n",
    "    )\n",
    "    revenue_growth: float = Field(description=\"Revenue growth percentage\")\n",
    "    gross_profit: Optional[float] = Field(None, description=\"Gross profit\")\n",
    "    gross_margin: float = Field(description=\"Gross margin percentage\")\n",
    "    operating_income: Optional[float] = Field(None, description=\"Operating income\")\n",
    "    operating_margin: Optional[float] = Field(\n",
    "        None, description=\"Operating margin percentage\"\n",
    "    )\n",
    "    net_income: float = Field(description=\"Net income\")\n",
    "    net_margin: Optional[float] = Field(None, description=\"Net margin percentage\")\n",
    "    eps: Optional[float] = Field(None, description=\"Basic earnings per share\")\n",
    "    diluted_eps: Optional[float] = Field(None, description=\"Diluted earnings per share\")\n",
    "    ebitda: Optional[float] = Field(\n",
    "        None,\n",
    "        description=\"EBITDA (Earnings Before Interest, Taxes, Depreciation, Amortization)\",\n",
    "    )\n",
    "    free_cash_flow: Optional[float] = Field(None, description=\"Free cash flow\")\n",
    "\n",
    "\n",
    "class BusinessSegment(BaseModel):\n",
    "    \"\"\"Information about a business segment\"\"\"\n",
    "\n",
    "    name: str = Field(description=\"Segment name\")\n",
    "    description: str = Field(description=\"Segment description\")\n",
    "    revenue: float = Field(None, description=\"Segment revenue\")\n",
    "    revenue_percentage: Optional[float] = Field(\n",
    "        None, description=\"Percentage of total company revenue\"\n",
    "    )\n",
    "    operating_income: Optional[float] = Field(\n",
    "        None, description=\"Segment operating income\"\n",
    "    )\n",
    "    operating_margin: Optional[float] = Field(\n",
    "        None, description=\"Segment operating margin percentage\"\n",
    "    )\n",
    "    year_over_year_growth: float = Field(\n",
    "        None, description=\"Year-over-year growth percentage\"\n",
    "    )\n",
    "    outlook: Optional[str] = Field(None, description=\"Future outlook for segment\")\n",
    "\n",
    "\n",
    "class GeographicSegment(BaseModel):\n",
    "    \"\"\"Information about a geographic segment\"\"\"\n",
    "\n",
    "    region: str = Field(description=\"Geographic region\")\n",
    "    revenue: float = Field(None, description=\"Revenue from region\")\n",
    "    revenue_percentage: Optional[float] = Field(\n",
    "        None, description=\"Percentage of total company revenue\"\n",
    "    )\n",
    "    year_over_year_growth: Optional[float] = Field(\n",
    "        None, description=\"Year-over-year growth percentage\"\n",
    "    )\n",
    "\n",
    "\n",
    "class RiskFactor(BaseModel):\n",
    "    \"\"\"Information about a risk factor\"\"\"\n",
    "\n",
    "    category: str = Field(\n",
    "        description=\"Risk category (e.g., Market, Operational, Legal)\"\n",
    "    )\n",
    "    title: Optional[str] = Field(None, description=\"Brief title of risk\")\n",
    "    description: str = Field(description=\"Description of risk factor\")\n",
    "    potential_impact: Optional[str] = Field(\n",
    "        None, description=\"Potential business impact\"\n",
    "    )\n",
    "\n",
    "\n",
    "class ManagementHighlights(BaseModel):\n",
    "    \"\"\"Key highlights from Management Discussion & Analysis\"\"\"\n",
    "\n",
    "    business_overview: str = Field(description=\"Overview of business and strategy\")\n",
    "    key_trends: Optional[str] = Field(\n",
    "        None, description=\"Key trends affecting performance\"\n",
    "    )\n",
    "    liquidity_assessment: Optional[str] = Field(\n",
    "        None, description=\"Management assessment of liquidity\"\n",
    "    )\n",
    "    outlook_summary: str = Field(description=\"Future outlook/guidance\")\n",
    "\n",
    "\n",
    "class SECFiling(BaseModel):\n",
    "    \"\"\"Schema for parsing 10-K and 10-Q filings from the SEC\"\"\"\n",
    "\n",
    "    filing_info: FilingInfo = Field(description=\"Basic information about the filing\")\n",
    "    company_profile: CompanyProfile = Field(description=\"Essential company information\")\n",
    "    financial_highlights: FinancialHighlights = Field(\n",
    "        description=\"Key financial metrics from this reporting period\"\n",
    "    )\n",
    "    business_segments: Optional[List[BusinessSegment]] = Field(\n",
    "        None, description=\"Key business segments information\"\n",
    "    )\n",
    "    geographic_segments: Optional[List[GeographicSegment]] = Field(\n",
    "        None, description=\"Geographic segment information\"\n",
    "    )\n",
    "    key_risks: List[RiskFactor] = Field(description=\"Most significant risk factors\")\n",
    "    mda_highlights: ManagementHighlights = Field(\n",
    "        description=\"Key highlights from Management Discussion & Analysis\"\n",
    "    )"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "9a0498e9",
   "metadata": {},
   "source": [
    "### 2. Extracting Information from $NVDA 10-K Filing\n",
    "\n",
    "Take a look at the schema definition above. We've defined a few models to represent the different sections of the 10K/Q filing. \n",
    "We've also defined a `SECFiling` model that combines all the sections into a single model. \n",
    "\n",
    "\n",
    "#### Design Considerations for Schema Definition\n",
    "\n",
    "- **Optional Fields**: There are quite a few optional fields in the schema. There are many fields that we would like to extract if present, but we know that they are not present in all filings. \n",
    "  e.g. companies which only has a US footprint will not have a geographic breakdown of their financials. It is important to designate these fields as optional so that the LLM is not \n",
    "  forced to make up values for these fields. Designating these fields as optional helps provide an escape hatch for the LLM to not hallucinate values for these fields. Note, however, that if aggressively marking fields as optional might result in the LLM being overly lazy and not attempt to extract information for these fields. So there's a balance in what fields to mark as optional! \n",
    "- **Descriptions for Fields**: While not mandatory, it is always a good idea to provide a description for each field. This helps the LLM understand the context in which the field is being extracted and can improve the accuracy of the extraction.  \n",
    "- **Enums**: We use enums to limit the possible values for a field. e.g. the `FilingInfo` model has an enum for the possible values of `filing_type`.  \n",
    "\n",
    "Now, let us create an agent to extract this information from the 10K/Q filing."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "2d335b32",
   "metadata": {},
   "outputs": [],
   "source": [
    "from llama_cloud.core.api_error import ApiError\n",
    "\n",
    "try:\n",
    "    existing_agent = llama_extract.get_agent(name=\"sec-10k-filing\")\n",
    "    if existing_agent:\n",
    "        llama_extract.delete_agent(existing_agent.id)\n",
    "except ApiError as e:\n",
    "    if e.status_code == 404:\n",
    "        pass\n",
    "    else:\n",
    "        raise\n",
    "\n",
    "agent = llama_extract.create_agent(name=\"sec-10k-filing\", data_schema=SECFiling)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "532f6ff5",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Uploading files: 100%|██████████| 1/1 [00:01<00:00,  1.14s/it]\n",
      "Creating extraction jobs: 100%|██████████| 1/1 [00:00<00:00,  2.78it/s]\n",
      "Extracting files: 100%|██████████| 1/1 [01:31<00:00, 91.56s/it]\n",
      "Uploading files: 100%|██████████| 1/1 [00:01<00:00,  1.26s/it]\n",
      "Creating extraction jobs: 100%|██████████| 1/1 [00:01<00:00,  1.44s/it]\n",
      "Extracting files: 100%|██████████| 1/1 [01:32<00:00, 92.73s/it]\n",
      "Uploading files: 100%|██████████| 1/1 [00:01<00:00,  1.14s/it]\n",
      "Creating extraction jobs: 100%|██████████| 1/1 [00:00<00:00,  2.85it/s]\n",
      "Extracting files: 100%|██████████| 1/1 [00:51<00:00, 51.87s/it]\n"
     ]
    }
   ],
   "source": [
    "nvda_10k_extract = agent.extract(\"./data/sec_filings/nvda_10k.pdf\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "83009725",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "{'filing_info': {'filing_type': '10-K',\n",
       "  'filing_date': '',\n",
       "  'reporting_period_end': '2025-01-26',\n",
       "  'fiscal_year': 2025,\n",
       "  'fiscal_quarter': 0},\n",
       " 'company_profile': {'name': 'NVIDIA Corporation',\n",
       "  'ticker': 'NVDA',\n",
       "  'reporting_currency': 'USD',\n",
       "  'exchanges': ['The Nasdaq Global Select Market'],\n",
       "  'auditor': None},\n",
       " 'financial_highlights': {'period_end': '2025-01-26',\n",
       "  'comparison_period_end': '2024-01-28',\n",
       "  'currency': 'USD',\n",
       "  'unit': 'thousands',\n",
       "  'revenue': 68038.0,\n",
       "  'revenue_prior_period': 26974.0,\n",
       "  'revenue_growth': 0.0,\n",
       "  'gross_profit': None,\n",
       "  'gross_margin': 75.0,\n",
       "  'operating_income': None,\n",
       "  'operating_margin': None,\n",
       "  'net_income': 72880.0,\n",
       "  'net_margin': None,\n",
       "  'eps': None,\n",
       "  'diluted_eps': None,\n",
       "  'ebitda': None,\n",
       "  'free_cash_flow': None},\n",
       " 'business_segments': [{'name': 'Compute & Networking',\n",
       "   'description': 'Strong demand for our accelerated computing and AI solutions. Revenue from Data Center computing grew 162% driven primarily by demand for our Hopper computing platform used for large language models, recommendation engines, and generative AI applications. Revenue from Data Center networking grew 51% driven by Ethernet for AI revenue, which includes Spectrum-X end-to-end ethernet platform.',\n",
       "   'revenue': 116193.0,\n",
       "   'revenue_percentage': 89.05,\n",
       "   'operating_income': 82875.0,\n",
       "   'operating_margin': 71.33,\n",
       "   'year_over_year_growth': 145.0,\n",
       "   'outlook': None},\n",
       "  {'name': 'Graphics',\n",
       "   'description': 'The year over year increase was driven by sales of our GeForce RTX 40 Series GPUs.',\n",
       "   'revenue': 14304.0,\n",
       "   'revenue_percentage': 10.95,\n",
       "   'operating_income': 5085.0,\n",
       "   'operating_margin': 35.55,\n",
       "   'year_over_year_growth': 6.0,\n",
       "   'outlook': None},\n",
       "  {'name': 'Data Center',\n",
       "   'description': 'Revenue by End Market',\n",
       "   'revenue': 115186.0,\n",
       "   'revenue_percentage': None,\n",
       "   'operating_income': None,\n",
       "   'operating_margin': None,\n",
       "   'year_over_year_growth': None,\n",
       "   'outlook': None},\n",
       "  {'name': 'Compute',\n",
       "   'description': 'Revenue by End Market',\n",
       "   'revenue': 102196.0,\n",
       "   'revenue_percentage': None,\n",
       "   'operating_income': None,\n",
       "   'operating_margin': None,\n",
       "   'year_over_year_growth': None,\n",
       "   'outlook': None},\n",
       "  {'name': 'Networking',\n",
       "   'description': 'Revenue by End Market',\n",
       "   'revenue': 12990.0,\n",
       "   'revenue_percentage': None,\n",
       "   'operating_income': None,\n",
       "   'operating_margin': None,\n",
       "   'year_over_year_growth': None,\n",
       "   'outlook': None},\n",
       "  {'name': 'Gaming',\n",
       "   'description': 'Revenue by End Market',\n",
       "   'revenue': 11350.0,\n",
       "   'revenue_percentage': None,\n",
       "   'operating_income': None,\n",
       "   'operating_margin': None,\n",
       "   'year_over_year_growth': None,\n",
       "   'outlook': None},\n",
       "  {'name': 'Professional Visualization',\n",
       "   'description': 'Revenue by End Market',\n",
       "   'revenue': 1878.0,\n",
       "   'revenue_percentage': None,\n",
       "   'operating_income': None,\n",
       "   'operating_margin': None,\n",
       "   'year_over_year_growth': None,\n",
       "   'outlook': None},\n",
       "  {'name': 'Automotive',\n",
       "   'description': 'Revenue by End Market',\n",
       "   'revenue': 1694.0,\n",
       "   'revenue_percentage': None,\n",
       "   'operating_income': None,\n",
       "   'operating_margin': None,\n",
       "   'year_over_year_growth': None,\n",
       "   'outlook': None},\n",
       "  {'name': 'OEM and Other',\n",
       "   'description': 'Revenue by End Market',\n",
       "   'revenue': 389.0,\n",
       "   'revenue_percentage': None,\n",
       "   'operating_income': None,\n",
       "   'operating_margin': None,\n",
       "   'year_over_year_growth': None,\n",
       "   'outlook': None}],\n",
       " 'geographic_segments': [{'region': 'Outside of the United States',\n",
       "   'revenue': None,\n",
       "   'revenue_percentage': 53.0,\n",
       "   'year_over_year_growth': -3.0},\n",
       "  {'region': 'United States',\n",
       "   'revenue': 61257.0,\n",
       "   'revenue_percentage': None,\n",
       "   'year_over_year_growth': None},\n",
       "  {'region': 'Singapore',\n",
       "   'revenue': 23684.0,\n",
       "   'revenue_percentage': 18.0,\n",
       "   'year_over_year_growth': None},\n",
       "  {'region': 'Taiwan',\n",
       "   'revenue': 20573.0,\n",
       "   'revenue_percentage': None,\n",
       "   'year_over_year_growth': None},\n",
       "  {'region': 'China (including Hong Kong)',\n",
       "   'revenue': 17108.0,\n",
       "   'revenue_percentage': None,\n",
       "   'year_over_year_growth': None},\n",
       "  {'region': 'Other',\n",
       "   'revenue': 7875.0,\n",
       "   'revenue_percentage': None,\n",
       "   'year_over_year_growth': None}],\n",
       " 'key_risks': [{'category': 'Operational',\n",
       "   'title': 'Supply-chain attacks or other business disruptions',\n",
       "   'description': 'We cannot guarantee that third parties and infrastructure in our supply chain or our partners’ supply chains have not been compromised or that they do not contain exploitable vulnerabilities, defects or bugs that could result in a breach of or disruption to our information technology systems, including our products and services, or the third-party information technology systems that support our services.',\n",
       "   'potential_impact': \"Potential reputational damage, regulatory scrutiny, or adverse impacts on the performance and reliability of our products, which could, in turn, affect our partners' operations, customer trust, and our revenue.\"},\n",
       "  {'category': 'Operational',\n",
       "   'title': \"Limited insight into third-party suppliers' data privacy or security practices\",\n",
       "   'description': 'Our ability to monitor these third parties’ information security practices is limited, and they may not have adequate information security measures in place.',\n",
       "   'potential_impact': 'If one of our third-party suppliers suffers a security incident, our response may be limited or more difficult because we may not have direct access to their systems, logs and other information related to the security incident.'},\n",
       "  {'category': 'Operational',\n",
       "   'title': 'Business disruptions',\n",
       "   'description': 'Business disruptions could harm our operations, lead to a decline in revenue and increase our costs. Factors that have caused and/or could in the future cause disruptions to our worldwide operations include: natural disasters, extreme weather conditions, power or water shortages, critical infrastructure failures, telecommunications failures, supplier disruptions, terrorist attacks, acts of violence, political and/or civil unrest, acts of war or other military actions, epidemics or pandemics, abrupt regulatory changes, and other natural or man-made disasters and catastrophic events.',\n",
       "   'potential_impact': 'Our operations vulnerable to natural disasters such as earthquakes, wildfires or other business disruptions occurring in these geographical areas. Catastrophic events can also have an impact on third-party vendors who provide us critical infrastructure services for IT and research and development systems and personnel. Geopolitical and domestic political developments and other events beyond our control can increase economic volatility globally. Political instability, changes in government or adverse political developments in or around any of the major countries in which we do business may harm our business, financial condition and results of operations.'},\n",
       "  {'category': 'Regulatory',\n",
       "   'title': 'Complex laws, rules, regulations, and political actions',\n",
       "   'description': 'We are subject to laws and regulations domestically and worldwide, affecting our operations in areas including, but not limited to, IP ownership and infringement; taxes; import and export requirements and tariffs; anti-corruption, including the Foreign Corrupt Practices Act; business acquisitions; foreign exchange controls and cash repatriation restrictions; foreign ownership and investment; data privacy requirements; competition and antitrust; advertising; employment; product regulations; cybersecurity; environmental, health, and safety requirements; the responsible use of AI; sustainability; cryptocurrency; and consumer laws.',\n",
       "   'potential_impact': 'Compliance with such requirements can be onerous and expensive, could impact our competitive position, and may negatively impact our business operations and ability to manufacture and ship our products. Violations could result in fines, criminal sanctions against us, our officers, or our employees, prohibitions on the conduct of our business, and damage to our reputation.'},\n",
       "  {'category': 'Regulatory',\n",
       "   'title': 'Export controls and geopolitical tensions',\n",
       "   'description': 'The USG announced export restrictions and export licensing requirements targeting China’s semiconductor and supercomputing industries. These restrictions impact exports of certain chips, as well as software, hardware, equipment and technology used to develop, produce and manufacture certain chips to China (including Hong Kong and Macau) and Russia, and specifically impact our A100 and H100 integrated circuits, DGX or any other systems or boards which incorporate A100 or H100 integrated circuits.',\n",
       "   'potential_impact': 'Such restrictions could increase the costs and burdens to us and our customers, delay or halt deployment of new systems using our products, and reduce the number of new entrants and customers, negatively impacting our business and financial results. Revisions to laws or regulations or their interpretation and enforcement could also result in increased taxation, trade sanctions, the imposition of or increase to import duties or tariffs, restrictions and controls on imports or exports, or other retaliatory actions, which could have an adverse effect on our business plans or impact the timing of our shipments.'},\n",
       "  {'category': 'Environmental',\n",
       "   'title': 'Climate change',\n",
       "   'description': 'Climate change may have an increasingly adverse impact on our business and on our customers, partners and vendors. Water and energy availability and reliability in the regions where we conduct business is critical, and certain of our facilities may be vulnerable to the impacts of extreme weather events.',\n",
       "   'potential_impact': 'Climate change, its impact on our supply chain and critical infrastructure worldwide and its potential to increase political instability in regions where we, our customers, partners and our vendors do business, may disrupt our business and cause us to experience higher attrition, losses and costs to maintain or resume operations. Losses not covered by insurance may be large, which could harm our results of operations and financial condition.'},\n",
       "  {'category': 'Regulatory',\n",
       "   'title': 'Chinese government restrictions',\n",
       "   'description': 'Restrictions imposed by the Chinese government on the duration of gaming activities and access to games may adversely affect our Gaming revenue, and increased oversight of digital platform companies may adversely affect our Data Center revenue. The Chinese government may also encourage customers to purchase from our China-based competitors, or impose restrictions on the sale to certain customers of our products, or any products containing components made by our partners and suppliers.',\n",
       "   'potential_impact': 'Negatively impact our business and financial results.'},\n",
       "  {'category': 'Operational',\n",
       "   'title': 'Supply chain disruptions',\n",
       "   'description': 'Our business depends on our ability to receive consistent and reliable supply from our overseas partners, especially in Taiwan and South Korea. Any new restrictions that negatively impact our ability to receive supply of components, parts, or services from Taiwan and South Korea, would negatively impact our business and financial results.',\n",
       "   'potential_impact': 'Negatively impact our business and financial results.'},\n",
       "  {'category': 'Reputational',\n",
       "   'title': 'Corporate sustainability practices scrutiny',\n",
       "   'description': 'Increased scrutiny from shareholders, regulators and others regarding our corporate sustainability practices could result in additional costs or risks and adversely impact our reputation and willingness of customers and suppliers to do business with us.',\n",
       "   'potential_impact': 'Negatively harm our brand, reputation and business activities or expose us to liability.'},\n",
       "  {'category': 'Reputational',\n",
       "   'title': 'Responsible use of AI technologies',\n",
       "   'description': 'Issues relating to the responsible use of our technologies, including AI in our offerings, may result in reputational or financial harm and liability. Concerns relating to the responsible use of new and evolving technologies, such as AI, in our products and services may result in reputational or financial harm and liability and may cause us to incur costs to resolve such issues.',\n",
       "   'potential_impact': 'Brand or reputational harm, competitive harm or legal liability.'},\n",
       "  {'category': 'Legal',\n",
       "   'title': 'Intellectual property rights protection',\n",
       "   'description': 'Actions to adequately protect our IP rights could result in substantial costs to us and our ability to compete could be harmed if we are unsuccessful or if we are prohibited from making or selling our products.',\n",
       "   'potential_impact': 'Increase our operating expenses and negatively impact our operating results.'},\n",
       "  {'category': 'Regulatory',\n",
       "   'title': 'Data privacy and security laws',\n",
       "   'description': 'We are subject to stringent and changing data privacy and security laws, rules, regulations and other obligations. These areas could damage our reputation, deter current and potential customers, affect our product design, or result in legal or regulatory proceedings and liability.',\n",
       "   'potential_impact': 'Material adverse effect on our reputation, business, or financial condition.'},\n",
       "  {'category': 'Regulatory',\n",
       "   'title': 'Tax liabilities and changes in tax laws',\n",
       "   'description': 'We may have exposure to additional tax liabilities and our operating results may be adversely impacted by changes in tax laws, higher than expected tax rates and other tax-related factors.',\n",
       "   'potential_impact': 'Adversely affect our provision for income taxes, cash tax payments, results of operations, and financial condition.'},\n",
       "  {'category': 'Legal',\n",
       "   'title': 'Litigation, investigations and regulatory proceedings',\n",
       "   'description': 'Our business is exposed to the burden and risks associated with litigation, investigations and regulatory proceedings.',\n",
       "   'potential_impact': 'Costly, time-consuming, and disruptive to our operations.'}],\n",
       " 'mda_highlights': {'business_overview': 'NVIDIA pioneered accelerated computing to help solve the most challenging computational problems. NVIDIA is now a full-stack computing infrastructure company with data-center-scale offerings that are reshaping industry. Our contracts may contain more than one performance obligation. Judgement is required in determining whether each performance obligation within a customer contract is distinct. Except for License and Development Arrangements, NVIDIA products and services function on a standalone basis and do not require a significant amount of integration or interdependency. Therefore, multiple performance obligations contained within a customer contract are considered distinct and are not combined for revenue recognition purposes.',\n",
       "  'key_trends': None,\n",
       "  'liquidity_assessment': None,\n",
       "  'outlook_summary': 'We believe that we have sufficient liquidity to meet our operating requirements for at least the next twelve months and thereafter for the foreseeable future, including our future supply obligations and share purchases. We continuously evaluate our liquidity and capital resources, including our access to external capital, to ensure we can finance future capital requirements.'}}"
      ]
     },
     "execution_count": null,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "nvda_10k_extract.data"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "b299a3e5",
   "metadata": {},
   "source": [
    "### 3. Assessing the Extraction Results\n",
    "\n",
    "Let's take a look at the extraction results for Nvidia's 10K filing. The description for management highlights and key risks looks reasonable at first glance. It is hard to verify the accuracy of the financial metrics since this is a long document with many pages.\n",
    "\n",
    "#### Adding Page Numbers to the Extraction Schema\n",
    "\n",
    "One way to make it easier to verify the accuracy of the extraction results is to add the page numbers to the extraction schema. This way, we can see which page numbers contain the key financial information. Let us add a `page_numbers` as a sub-field to `FinancialHighlights`, `BusinessSegment` and `GeographicSegment` fields to make it easier for us to verify key financial metrics extracted. \n",
    "\n",
    "> **Note**: Page numbers might be off by one due to the relative placement of the page numbers and the surrounding context from which the information is extracted, but it is a quick way to navigate to the relevant sections of the document and sanity test some fields.\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d20eca24",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "{'$defs': {'BusinessSegment': {'description': 'Information about a business segment',\n",
       "   'properties': {'name': {'description': 'Segment name',\n",
       "     'title': 'Name',\n",
       "     'type': 'string'},\n",
       "    'description': {'description': 'Segment description',\n",
       "     'title': 'Description',\n",
       "     'type': 'string'},\n",
       "    'revenue': {'default': None,\n",
       "     'description': 'Segment revenue',\n",
       "     'title': 'Revenue',\n",
       "     'type': 'number'},\n",
       "    'revenue_percentage': {'anyOf': [{'type': 'number'}, {'type': 'null'}],\n",
       "     'default': None,\n",
       "     'description': 'Percentage of total company revenue',\n",
       "     'title': 'Revenue Percentage'},\n",
       "    'operating_income': {'anyOf': [{'type': 'number'}, {'type': 'null'}],\n",
       "     'default': None,\n",
       "     'description': 'Segment operating income',\n",
       "     'title': 'Operating Income'},\n",
       "    'operating_margin': {'anyOf': [{'type': 'number'}, {'type': 'null'}],\n",
       "     'default': None,\n",
       "     'description': 'Segment operating margin percentage',\n",
       "     'title': 'Operating Margin'},\n",
       "    'year_over_year_growth': {'default': None,\n",
       "     'description': 'Year-over-year growth percentage',\n",
       "     'title': 'Year Over Year Growth',\n",
       "     'type': 'number'},\n",
       "    'outlook': {'anyOf': [{'type': 'string'}, {'type': 'null'}],\n",
       "     'default': None,\n",
       "     'description': 'Future outlook for segment',\n",
       "     'title': 'Outlook'},\n",
       "    'page_numbers': {'description': 'Page numbers (at bottom of the page) where the financial metrics above are extracted from.',\n",
       "     'items': {'type': 'integer'},\n",
       "     'title': 'Page Numbers',\n",
       "     'type': 'array'}},\n",
       "   'required': ['name', 'description', 'page_numbers'],\n",
       "   'title': 'BusinessSegment',\n",
       "   'type': 'object'},\n",
       "  'CompanyProfile': {'description': 'Essential company information',\n",
       "   'properties': {'name': {'description': 'Legal name of company',\n",
       "     'title': 'Name',\n",
       "     'type': 'string'},\n",
       "    'ticker': {'description': 'Stock ticker symbol',\n",
       "     'title': 'Ticker',\n",
       "     'type': 'string'},\n",
       "    'reporting_currency': {'description': 'Currency used in financial statements',\n",
       "     'title': 'Reporting Currency',\n",
       "     'type': 'string'},\n",
       "    'exchanges': {'anyOf': [{'items': {'type': 'string'}, 'type': 'array'},\n",
       "      {'type': 'null'}],\n",
       "     'default': None,\n",
       "     'description': 'Stock exchanges where listed',\n",
       "     'title': 'Exchanges'},\n",
       "    'auditor': {'anyOf': [{'type': 'string'}, {'type': 'null'}],\n",
       "     'default': None,\n",
       "     'description': \"Company's auditor\",\n",
       "     'title': 'Auditor'}},\n",
       "   'required': ['name', 'ticker', 'reporting_currency'],\n",
       "   'title': 'CompanyProfile',\n",
       "   'type': 'object'},\n",
       "  'FilingInfo': {'description': 'Basic information about the SEC filing',\n",
       "   'properties': {'filing_type': {'description': 'Type of SEC filing',\n",
       "     'enum': ['10-K', '10-Q', '10-K/A', '10-Q/A'],\n",
       "     'title': 'Filing Type',\n",
       "     'type': 'string'},\n",
       "    'filing_date': {'description': 'Date when filing was submitted to SEC',\n",
       "     'title': 'Filing Date',\n",
       "     'type': 'string'},\n",
       "    'reporting_period_end': {'description': 'End date of reporting period',\n",
       "     'title': 'Reporting Period End',\n",
       "     'type': 'string'},\n",
       "    'fiscal_year': {'description': 'Fiscal year',\n",
       "     'title': 'Fiscal Year',\n",
       "     'type': 'integer'},\n",
       "    'fiscal_quarter': {'description': 'Fiscal quarter (if 10-Q)',\n",
       "     'maximum': 4,\n",
       "     'minimum': 1,\n",
       "     'title': 'Fiscal Quarter',\n",
       "     'type': 'integer'}},\n",
       "   'required': ['filing_type',\n",
       "    'filing_date',\n",
       "    'reporting_period_end',\n",
       "    'fiscal_year',\n",
       "    'fiscal_quarter'],\n",
       "   'title': 'FilingInfo',\n",
       "   'type': 'object'},\n",
       "  'FinancialHighlights': {'description': 'Key financial metrics from this reporting period',\n",
       "   'properties': {'period_end': {'description': 'End date of reporting period',\n",
       "     'title': 'Period End',\n",
       "     'type': 'string'},\n",
       "    'comparison_period_end': {'anyOf': [{'type': 'string'}, {'type': 'null'}],\n",
       "     'default': None,\n",
       "     'description': 'End date of comparison period (typically prior year/quarter)',\n",
       "     'title': 'Comparison Period End'},\n",
       "    'currency': {'description': 'Currency of financial figures',\n",
       "     'title': 'Currency',\n",
       "     'type': 'string'},\n",
       "    'unit': {'description': 'Unit of financial figures (thousands, millions, etc.)',\n",
       "     'title': 'Unit',\n",
       "     'type': 'string'},\n",
       "    'revenue': {'description': 'Total revenue for period',\n",
       "     'title': 'Revenue',\n",
       "     'type': 'number'},\n",
       "    'revenue_prior_period': {'anyOf': [{'type': 'number'}, {'type': 'null'}],\n",
       "     'default': None,\n",
       "     'description': 'Revenue from comparison period',\n",
       "     'title': 'Revenue Prior Period'},\n",
       "    'revenue_growth': {'description': 'Revenue growth percentage',\n",
       "     'title': 'Revenue Growth',\n",
       "     'type': 'number'},\n",
       "    'gross_profit': {'anyOf': [{'type': 'number'}, {'type': 'null'}],\n",
       "     'default': None,\n",
       "     'description': 'Gross profit',\n",
       "     'title': 'Gross Profit'},\n",
       "    'gross_margin': {'description': 'Gross margin percentage',\n",
       "     'title': 'Gross Margin',\n",
       "     'type': 'number'},\n",
       "    'operating_income': {'anyOf': [{'type': 'number'}, {'type': 'null'}],\n",
       "     'default': None,\n",
       "     'description': 'Operating income',\n",
       "     'title': 'Operating Income'},\n",
       "    'operating_margin': {'anyOf': [{'type': 'number'}, {'type': 'null'}],\n",
       "     'default': None,\n",
       "     'description': 'Operating margin percentage',\n",
       "     'title': 'Operating Margin'},\n",
       "    'net_income': {'description': 'Net income',\n",
       "     'title': 'Net Income',\n",
       "     'type': 'number'},\n",
       "    'net_margin': {'anyOf': [{'type': 'number'}, {'type': 'null'}],\n",
       "     'default': None,\n",
       "     'description': 'Net margin percentage',\n",
       "     'title': 'Net Margin'},\n",
       "    'eps': {'anyOf': [{'type': 'number'}, {'type': 'null'}],\n",
       "     'default': None,\n",
       "     'description': 'Basic earnings per share',\n",
       "     'title': 'Eps'},\n",
       "    'diluted_eps': {'anyOf': [{'type': 'number'}, {'type': 'null'}],\n",
       "     'default': None,\n",
       "     'description': 'Diluted earnings per share',\n",
       "     'title': 'Diluted Eps'},\n",
       "    'ebitda': {'anyOf': [{'type': 'number'}, {'type': 'null'}],\n",
       "     'default': None,\n",
       "     'description': 'EBITDA (Earnings Before Interest, Taxes, Depreciation, Amortization)',\n",
       "     'title': 'Ebitda'},\n",
       "    'free_cash_flow': {'anyOf': [{'type': 'number'}, {'type': 'null'}],\n",
       "     'default': None,\n",
       "     'description': 'Free cash flow',\n",
       "     'title': 'Free Cash Flow'},\n",
       "    'page_numbers': {'description': 'Page numbers (at bottom of the page) where the financial metrics above are extracted from.',\n",
       "     'items': {'type': 'integer'},\n",
       "     'title': 'Page Numbers',\n",
       "     'type': 'array'}},\n",
       "   'required': ['period_end',\n",
       "    'currency',\n",
       "    'unit',\n",
       "    'revenue',\n",
       "    'revenue_growth',\n",
       "    'gross_margin',\n",
       "    'net_income',\n",
       "    'page_numbers'],\n",
       "   'title': 'FinancialHighlights',\n",
       "   'type': 'object'},\n",
       "  'GeographicSegment': {'description': 'Information about a geographic segment',\n",
       "   'properties': {'region': {'description': 'Geographic region',\n",
       "     'title': 'Region',\n",
       "     'type': 'string'},\n",
       "    'revenue': {'default': None,\n",
       "     'description': 'Revenue from region',\n",
       "     'title': 'Revenue',\n",
       "     'type': 'number'},\n",
       "    'revenue_percentage': {'anyOf': [{'type': 'number'}, {'type': 'null'}],\n",
       "     'default': None,\n",
       "     'description': 'Percentage of total company revenue',\n",
       "     'title': 'Revenue Percentage'},\n",
       "    'year_over_year_growth': {'anyOf': [{'type': 'number'}, {'type': 'null'}],\n",
       "     'default': None,\n",
       "     'description': 'Year-over-year growth percentage',\n",
       "     'title': 'Year Over Year Growth'},\n",
       "    'page_numbers': {'description': 'Page numbers (at bottom of the page) where the financial metrics above are extracted from.',\n",
       "     'items': {'type': 'integer'},\n",
       "     'title': 'Page Numbers',\n",
       "     'type': 'array'}},\n",
       "   'required': ['region', 'page_numbers'],\n",
       "   'title': 'GeographicSegment',\n",
       "   'type': 'object'},\n",
       "  'ManagementHighlights': {'description': 'Key highlights from Management Discussion & Analysis',\n",
       "   'properties': {'business_overview': {'description': 'Overview of business and strategy',\n",
       "     'title': 'Business Overview',\n",
       "     'type': 'string'},\n",
       "    'key_trends': {'anyOf': [{'type': 'string'}, {'type': 'null'}],\n",
       "     'default': None,\n",
       "     'description': 'Key trends affecting performance',\n",
       "     'title': 'Key Trends'},\n",
       "    'liquidity_assessment': {'anyOf': [{'type': 'string'}, {'type': 'null'}],\n",
       "     'default': None,\n",
       "     'description': 'Management assessment of liquidity',\n",
       "     'title': 'Liquidity Assessment'},\n",
       "    'outlook_summary': {'description': 'Future outlook/guidance',\n",
       "     'title': 'Outlook Summary',\n",
       "     'type': 'string'}},\n",
       "   'required': ['business_overview', 'outlook_summary'],\n",
       "   'title': 'ManagementHighlights',\n",
       "   'type': 'object'},\n",
       "  'RiskFactor': {'description': 'Information about a risk factor',\n",
       "   'properties': {'category': {'description': 'Risk category (e.g., Market, Operational, Legal)',\n",
       "     'title': 'Category',\n",
       "     'type': 'string'},\n",
       "    'title': {'anyOf': [{'type': 'string'}, {'type': 'null'}],\n",
       "     'default': None,\n",
       "     'description': 'Brief title of risk',\n",
       "     'title': 'Title'},\n",
       "    'description': {'description': 'Description of risk factor',\n",
       "     'title': 'Description',\n",
       "     'type': 'string'},\n",
       "    'potential_impact': {'anyOf': [{'type': 'string'}, {'type': 'null'}],\n",
       "     'default': None,\n",
       "     'description': 'Potential business impact',\n",
       "     'title': 'Potential Impact'}},\n",
       "   'required': ['category', 'description'],\n",
       "   'title': 'RiskFactor',\n",
       "   'type': 'object'}},\n",
       " 'description': 'Schema for parsing 10-K and 10-Q filings from the SEC',\n",
       " 'properties': {'filing_info': {'$ref': '#/$defs/FilingInfo',\n",
       "   'description': 'Basic information about the filing'},\n",
       "  'company_profile': {'$ref': '#/$defs/CompanyProfile'},\n",
       "  'financial_highlights': {'$ref': '#/$defs/FinancialHighlights'},\n",
       "  'business_segments': {'anyOf': [{'items': {'$ref': '#/$defs/BusinessSegment'},\n",
       "     'type': 'array'},\n",
       "    {'type': 'null'}],\n",
       "   'default': None,\n",
       "   'description': 'Key business segments information',\n",
       "   'title': 'Business Segments'},\n",
       "  'geographic_segments': {'anyOf': [{'items': {'$ref': '#/$defs/GeographicSegment'},\n",
       "     'type': 'array'},\n",
       "    {'type': 'null'}],\n",
       "   'default': None,\n",
       "   'description': 'Geographic segment information',\n",
       "   'title': 'Geographic Segments'},\n",
       "  'key_risks': {'description': 'Most significant risk factors',\n",
       "   'items': {'$ref': '#/$defs/RiskFactor'},\n",
       "   'title': 'Key Risks',\n",
       "   'type': 'array'},\n",
       "  'mda_highlights': {'$ref': '#/$defs/ManagementHighlights'}},\n",
       " 'required': ['filing_info',\n",
       "  'company_profile',\n",
       "  'financial_highlights',\n",
       "  'key_risks',\n",
       "  'mda_highlights'],\n",
       " 'title': 'SECFiling',\n",
       " 'type': 'object'}"
      ]
     },
     "execution_count": null,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from pydantic.fields import FieldInfo\n",
    "\n",
    "FinancialHighlights.__annotations__[\"page_numbers\"] = List[int]\n",
    "FinancialHighlights.model_fields[\"page_numbers\"] = FieldInfo(\n",
    "    annotation=List[int],\n",
    "    description=\"Page numbers (at bottom of the page) where the financial metrics above are extracted from.\",\n",
    ")\n",
    "FinancialHighlights.model_rebuild(force=True)\n",
    "\n",
    "BusinessSegment.model_fields[\"page_numbers\"] = FieldInfo(\n",
    "    annotation=List[int],\n",
    "    description=\"Page numbers (at bottom of the page) where the financial metrics above are extracted from.\",\n",
    ")\n",
    "BusinessSegment.model_rebuild(force=True)\n",
    "\n",
    "GeographicSegment.model_fields[\"page_numbers\"] = FieldInfo(\n",
    "    annotation=List[int],\n",
    "    description=\"Page numbers (at bottom of the page) where the financial metrics above are extracted from.\",\n",
    ")\n",
    "GeographicSegment.model_rebuild(force=True)\n",
    "\n",
    "SECFiling.model_rebuild(force=True)\n",
    "SECFiling.model_json_schema()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "0fcd3adb",
   "metadata": {},
   "outputs": [],
   "source": [
    "agent.data_schema = SECFiling"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "3ccd0a19",
   "metadata": {},
   "outputs": [],
   "source": [
    "nvda_10k_extract = agent.extract(\"./data/sec_filings/nvda_10k.pdf\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "a733774b",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "{'filing_info': {'filing_type': '10-K',\n",
       "  'filing_date': '2025-01-26',\n",
       "  'reporting_period_end': '2025-01-26',\n",
       "  'fiscal_year': 2025,\n",
       "  'fiscal_quarter': 1},\n",
       " 'company_profile': {'name': 'NVIDIA Corporation',\n",
       "  'ticker': 'NVDA',\n",
       "  'reporting_currency': 'USD',\n",
       "  'exchanges': ['The Nasdaq Global Select Market'],\n",
       "  'auditor': None},\n",
       " 'financial_highlights': {'period_end': '2025-01-26',\n",
       "  'comparison_period_end': '2024-01-28',\n",
       "  'currency': 'USD',\n",
       "  'unit': 'thousands',\n",
       "  'revenue': 130497.0,\n",
       "  'revenue_prior_period': 60922.0,\n",
       "  'revenue_growth': 114.23,\n",
       "  'gross_profit': 97858.0,\n",
       "  'gross_margin': 75.0,\n",
       "  'operating_income': 81453.0,\n",
       "  'operating_margin': None,\n",
       "  'net_income': 72880.0,\n",
       "  'net_margin': 55.8,\n",
       "  'eps': None,\n",
       "  'diluted_eps': None,\n",
       "  'ebitda': None,\n",
       "  'free_cash_flow': None,\n",
       "  'page_numbers': [40, 41, 55, 56, 68]},\n",
       " 'business_segments': [{'name': 'Compute & Networking',\n",
       "   'description': 'Includes Data Center accelerated computing platforms and AI solutions and software; networking; automotive platforms and autonomous and electric vehicle solutions; Jetson for robotics and other embedded platforms; and DGX Cloud computing services. Strong demand for accelerated computing and AI solutions. Revenue from Data Center computing grew 162% driven primarily by demand for our Hopper computing platform used for large language models, recommendation engines, and generative AI applications. Revenue from Data Center networking grew 51% driven by Ethernet for AI revenue, which includes Spectrum-X end-to-end ethernet platform. Includes product costs and inventory provisions, compensation and benefits excluding stock-based compensation expense, compute and infrastructure expenses, and engineering development costs.',\n",
       "   'revenue': 116193.0,\n",
       "   'revenue_percentage': 88.99,\n",
       "   'operating_income': 82875.0,\n",
       "   'operating_margin': 71.3,\n",
       "   'year_over_year_growth': 145.0,\n",
       "   'outlook': 'Higher U.S.-based Compute & Networking segment demand.',\n",
       "   'page_numbers': [5, 40, 68, 79]},\n",
       "  {'name': 'Graphics',\n",
       "   'description': 'Includes GeForce GPUs for gaming and PCs, the GeForce NOW game streaming service and related infrastructure, and solutions for gaming platforms; Quadro/NVIDIA RTX GPUs for enterprise workstation graphics; virtual GPU, or vGPU, software for cloud-based visual and virtual computing; automotive platforms for infotainment systems; and Omniverse Enterprise software for building and operating industrial AI and digital twin applications. The year over year increase was driven by sales of our GeForce RTX 40 Series GPUs. Includes product costs and inventory provisions, compensation and benefits excluding stock-based compensation expense, compute and infrastructure expenses, and engineering development costs.',\n",
       "   'revenue': 14304.0,\n",
       "   'revenue_percentage': 11.0,\n",
       "   'operating_income': 5085.0,\n",
       "   'operating_margin': 35.6,\n",
       "   'year_over_year_growth': 6.0,\n",
       "   'outlook': None,\n",
       "   'page_numbers': [5, 40, 68, 79]}],\n",
       " 'geographic_segments': [{'region': 'Outside of the United States',\n",
       "   'revenue': None,\n",
       "   'revenue_percentage': 53.0,\n",
       "   'year_over_year_growth': None,\n",
       "   'page_numbers': [42]},\n",
       "  {'region': 'United States',\n",
       "   'revenue': 61257.0,\n",
       "   'revenue_percentage': None,\n",
       "   'year_over_year_growth': None,\n",
       "   'page_numbers': [79]},\n",
       "  {'region': 'Singapore',\n",
       "   'revenue': 23684.0,\n",
       "   'revenue_percentage': 18.0,\n",
       "   'year_over_year_growth': None,\n",
       "   'page_numbers': [79]},\n",
       "  {'region': 'Taiwan',\n",
       "   'revenue': 20573.0,\n",
       "   'revenue_percentage': None,\n",
       "   'year_over_year_growth': None,\n",
       "   'page_numbers': [79]},\n",
       "  {'region': 'China (including Hong Kong)',\n",
       "   'revenue': 17108.0,\n",
       "   'revenue_percentage': None,\n",
       "   'year_over_year_growth': None,\n",
       "   'page_numbers': [79]},\n",
       "  {'region': 'Other',\n",
       "   'revenue': 7875.0,\n",
       "   'revenue_percentage': None,\n",
       "   'year_over_year_growth': None,\n",
       "   'page_numbers': [79]}],\n",
       " 'key_risks': [{'category': 'Regulatory, Legal, Our Stock, and Other Matters',\n",
       "   'title': 'Risks Related to Regulatory, Legal, Our Stock, and Other Matters',\n",
       "   'description': 'We are subject to complex laws, rules, regulations, and political and other actions, including restrictions on the export of our products, which may adversely impact our business.',\n",
       "   'potential_impact': None},\n",
       "  {'category': 'Regulatory, Legal',\n",
       "   'title': 'Increased scrutiny regarding our corporate sustainability practices could result in financial, reputational, or operational harm and liability.',\n",
       "   'description': 'Increased scrutiny regarding our corporate sustainability practices could result in financial, reputational, or operational harm and liability.',\n",
       "   'potential_impact': None},\n",
       "  {'category': 'Regulatory',\n",
       "   'title': 'Issues relating to the responsible use of our technologies, including AI',\n",
       "   'description': 'Issues relating to the responsible use of our technologies, including AI, may result in reputational or financial harm and liability.',\n",
       "   'potential_impact': None},\n",
       "  {'category': 'Operational',\n",
       "   'title': 'Supply-chain attacks or other business disruptions',\n",
       "   'description': \"We cannot guarantee that third parties and infrastructure in our supply chain or our partners’ supply chains have not been compromised or that they do not contain exploitable vulnerabilities, defects or bugs that could result in a breach of or disruption to our information technology systems, including our products and services, or the third-party information technology systems that support our services. We have incorporated third-party data into some of our AI models and used open-source datasets to train our models and may continue to do so. These datasets may be flawed, insufficient, or contain certain biased information, and may otherwise decrease resilience to security incidents that may compromise the integrity of our AI outputs, leading to potential reputational damage, regulatory scrutiny, or adverse impacts on the performance and reliability of our products, which could, in turn, affect our partners' operations, customer trust, and our revenue.\",\n",
       "   'potential_impact': \"Potential reputational damage, regulatory scrutiny, or adverse impacts on the performance and reliability of our products, which could, in turn, affect our partners' operations, customer trust, and our revenue.\"},\n",
       "  {'category': 'Operational',\n",
       "   'title': 'Limited insight into data privacy or security practices of third-party suppliers',\n",
       "   'description': 'Our ability to monitor these third parties’ information security practices is limited, and they may not have adequate information security measures in place. In addition, if one of our third-party suppliers suffers a security incident (which has happened in the past and may happen in the future), our response may be limited or more difficult because we may not have direct access to their systems, logs and other information related to the security incident.',\n",
       "   'potential_impact': 'Potential liability and harm to our business if our products or services are compromised, affecting a significant number of our customers and their data.'},\n",
       "  {'category': 'Operational',\n",
       "   'title': 'Business disruptions',\n",
       "   'description': 'Business disruptions could harm our operations, lead to a decline in revenue and increase our costs. Factors that have caused and/or could in the future cause disruptions to our worldwide operations include: natural disasters, extreme weather conditions, power or water shortages, critical infrastructure failures, telecommunications failures, supplier disruptions, terrorist attacks, acts of violence, political and/or civil unrest, acts of war or other military actions, epidemics or pandemics, abrupt regulatory changes, and other natural or man-made disasters and catastrophic events.',\n",
       "   'potential_impact': 'Harm to our operations, lead to a decline in revenue and increase our costs.'},\n",
       "  {'category': 'Operational',\n",
       "   'title': 'Geopolitical tensions and conflicts',\n",
       "   'description': 'Worldwide geopolitical tensions and conflicts, including but not limited to China, Hong Kong, Israel, Korea and Taiwan where the manufacture of our product components and final assembly of our products are concentrated may result in changing regulatory requirements, and other disruptions that could impact our operations and operating strategies, product demand, access to global markets, hiring, and profitability.',\n",
       "   'potential_impact': 'Our operations could be harmed and our costs could increase if manufacturing, logistics, or other operations are disrupted for any reason, including natural disasters, high heat events, water shortages, power shortages, information technology system failures or cyber-attacks, military actions or economic, and business, labor, environmental, public health, or political issues.'},\n",
       "  {'category': 'Operational',\n",
       "   'title': 'Interruptions or delays in services from CSPs, data center co-location partners, and other third parties',\n",
       "   'description': 'Interruptions or delays in services from CSPs, data center co-location partners, and other third parties on which we rely, including due to the events described above or other events such as the insolvency of these parties, could impair our ability to provide our products and services and harm our business.',\n",
       "   'potential_impact': 'Impair our ability to provide our products and services and harm our business.'},\n",
       "  {'category': 'Environmental',\n",
       "   'title': 'Climate change',\n",
       "   'description': 'Climate change may have an increasingly adverse impact on our business and on our customers, partners and vendors. Water and energy availability and reliability in the regions where we conduct business is critical, and certain of our facilities may be vulnerable to the impacts of extreme weather events.',\n",
       "   'potential_impact': 'Disrupt our business and cause us to experience higher attrition, losses and costs to maintain or resume operations. Losses not covered by insurance may be large, which could harm our results of operations and financial condition.'},\n",
       "  {'category': 'Regulatory',\n",
       "   'title': 'Export controls and geopolitical tensions',\n",
       "   'description': 'The USG announced export restrictions and export licensing requirements targeting China’s semiconductor and supercomputing industries. These restrictions impact exports of certain chips, as well as software, hardware, equipment and technology used to develop, produce and manufacture our products.',\n",
       "   'potential_impact': 'Could increase the costs and burdens to us and our customers, delay or halt deployment of new systems using our products, and reduce the number of new entrants and customers, negatively impacting our business and financial results.'},\n",
       "  {'category': 'Regulatory',\n",
       "   'title': 'Chinese government restrictions',\n",
       "   'description': 'Restrictions imposed by the Chinese government on the duration of gaming activities and access to games may adversely affect our Gaming revenue, and increased oversight of digital platform companies may adversely affect our Data Center revenue. The Chinese government may also encourage customers to purchase from our China-based competitors, or impose restrictions on the sale to certain customers of our products, or any products containing components made by our partners and suppliers.',\n",
       "   'potential_impact': 'Negatively impact our business and financial results.'},\n",
       "  {'category': 'Operational',\n",
       "   'title': 'Supply chain disruptions',\n",
       "   'description': 'Our business depends on our ability to receive consistent and reliable supply from our overseas partners, especially in Taiwan and South Korea. Any new restrictions that negatively impact our ability to receive supply of components, parts, or services from Taiwan and South Korea, would negatively impact our business and financial results.',\n",
       "   'potential_impact': 'Negatively impact our business and financial results.'},\n",
       "  {'category': 'Reputational',\n",
       "   'title': 'Corporate sustainability practices scrutiny',\n",
       "   'description': 'Increased scrutiny from shareholders, regulators and others regarding our corporate sustainability practices could result in additional costs or risks and adversely impact our reputation and willingness of customers and suppliers to do business with us.',\n",
       "   'potential_impact': 'Negatively harm our brand, reputation and business activities or expose us to liability.'},\n",
       "  {'category': 'Reputational/Legal',\n",
       "   'title': 'Responsible use of AI technologies',\n",
       "   'description': 'Issues relating to the responsible use of our technologies, including AI in our offerings, may result in reputational or financial harm and liability. Concerns relating to the responsible use of new and evolving technologies, such as AI, in our products and services may result in reputational or financial harm and liability and may cause us to incur costs to resolve such issues.',\n",
       "   'potential_impact': 'Reputational or financial harm and liability.'},\n",
       "  {'category': 'Legal',\n",
       "   'title': 'Intellectual property rights protection',\n",
       "   'description': 'Actions to adequately protect our IP rights could result in substantial costs to us and our ability to compete could be harmed if we are unsuccessful or if we are prohibited from making or selling our products.',\n",
       "   'potential_impact': 'Our business could be negatively impacted.'},\n",
       "  {'category': 'Regulatory',\n",
       "   'title': 'Data privacy and security laws',\n",
       "   'description': 'We are subject to stringent and changing data privacy and security laws, rules, regulations and other obligations. These areas could damage our reputation, deter current and potential customers, affect our product design, or result in legal or regulatory proceedings and liability.',\n",
       "   'potential_impact': 'Material adverse effect on our reputation, business, or financial condition.'},\n",
       "  {'category': 'Legal/Financial',\n",
       "   'title': 'Tax liabilities and changes in tax laws',\n",
       "   'description': 'We may have exposure to additional tax liabilities and our operating results may be adversely impacted by changes in tax laws, higher than expected tax rates and other tax-related factors.',\n",
       "   'potential_impact': 'Adversely affect our provision for income taxes, cash tax payments, results of operations, and financial condition.'},\n",
       "  {'category': 'Legal',\n",
       "   'title': 'Litigation, investigations and regulatory proceedings',\n",
       "   'description': 'Our business is exposed to the burden and risks associated with litigation, investigations and regulatory proceedings.',\n",
       "   'potential_impact': 'Litigation can be costly, time-consuming, and disruptive to our operations.'},\n",
       "  {'category': 'Legal',\n",
       "   'title': 'Securities Class Action and Derivative Lawsuits',\n",
       "   'description': 'The plaintiffs in the putative securities class action lawsuit, captioned 4:18-cv-07669-HSG, initially filed on December 21, 2018 in the United States District Court for the Northern District of California, and titled In Re NVIDIA Corporation Securities Litigation, filed an amended complaint on May 13, 2020. The amended complaint asserted that NVIDIA and certain NVIDIA executives violated Section 10(b) of the Securities Exchange Act of 1934, as amended, or the Exchange Act, and SEC Rule 10b-5, by making materially false or misleading statements related to channel inventory and the impact of cryptocurrency mining on GPU demand between May 10, 2017 and November 14, 2018. Plaintiffs also alleged that the NVIDIA executives who they named as defendants violated Section 20(a) of the Exchange Act. Plaintiffs sought class certification, an award of unspecified compensatory damages, an award of reasonable costs and expenses, including attorneys’ fees and expert fees, and further relief as the Court may deem just and proper.',\n",
       "   'potential_impact': 'Unspecified damages and other relief, including reforms and improvements to NVIDIA’s corporate governance and internal procedures.'},\n",
       "  {'category': 'Legal',\n",
       "   'title': 'Insider trading restrictions',\n",
       "   'description': 'You may be subject to insider trading restrictions and/or market abuse laws based on the exchange on which the shares of Common Stock are listed and in applicable jurisdictions, including the United States and your country or your broker’s country, if different, which may affect your ability to accept, acquire, sell or otherwise dispose of shares of Common Stock, rights to shares of Common Stock (e.g., Restricted Stock Units) or rights linked to the value of shares of Common Stock during such times as you are considered to have “inside information” regarding the Company (as defined by the laws in applicable jurisdictions). Local insider trading laws and regulations may prohibit the cancellation or amendment of orders you placed before you possessed inside information. Furthermore, you could be prohibited from (i) disclosing the inside information to any third party, which may include fellow employees and (ii) “tipping” third parties or causing them otherwise to buy or sell securities. Any restrictions under these laws or regulations are separate from and in addition to any restrictions that may be imposed under any applicable insider trading policy of the Company.',\n",
       "   'potential_impact': 'Affect your ability to accept, acquire, sell or otherwise dispose of shares of Common Stock, rights to shares of Common Stock (e.g., Restricted Stock Units) or rights linked to the value of shares of Common Stock during such times as you are considered to have “inside information” regarding the Company.'}],\n",
       " 'mda_highlights': {'business_overview': 'NVIDIA pioneered accelerated computing to help solve the most challenging computational problems. NVIDIA is now a full-stack computing infrastructure company with data-center-scale offerings that are reshaping industry. NVIDIA invents computing technologies that improve lives and address global challenges. Our goal is to integrate sound environmental, social, and corporate governance principles and practices into every aspect of the Company. Headquartered in Santa Clara, California, NVIDIA was incorporated in California in April 1993 and reincorporated in Delaware in April 1998. We refer to customers who purchase products directly from NVIDIA as direct customers, such as AIBs, distributors, ODMs, OEMs, and system integrators. The number of Restricted Stock Units (and the related shares of Common Stock) subject to your Award will be adjusted from time to time for Capitalization Adjustments, as provided in the Plan.',\n",
       "  'key_trends': None,\n",
       "  'liquidity_assessment': 'We believe that we have sufficient liquidity to meet our operating requirements for at least the next twelve months and thereafter for the foreseeable future, including our future supply obligations and share purchases. We continuously evaluate our liquidity and capital resources, including our access to external capital, to ensure we can finance future capital requirements.',\n",
       "  'outlook_summary': 'NVIDIA has a platform strategy, bringing together hardware, systems, software, algorithms, libraries, and services to create unique value for the markets we serve. While the computing requirements of these end markets are diverse, we address them with a unified underlying architecture leveraging our GPUs and networking and software stacks. The programmable nature of our architecture allows us to support several multi-billion-dollar end markets with the same underlying technology by using a variety of software stacks developed either internally or by third-party developers and partners. The large and growing number of developers and installed base across our platforms strengthens our ecosystem and increases the value of our platform to our customers. We committed to purchase or generate enough renewable energy to match 100% of our global electricity usage for offices and data centers under our operational control starting with our fiscal year 2025. In fiscal year 2024, we made progress towards this goal and increased the percentage of our electricity use matched by renewable energy to 76%. By the end of fiscal year 2026, we also aim to engage manufacturing suppliers comprising at least 67% of NVIDIA’s scope 3 category 1 GHG emissions with the goal of effecting supplier adoption of science-based targets. As of January 26, 2025, revenue related to remaining performance obligations from contracts greater than one year in length was $1.7 billion, which includes $1.6 billion from deferred revenue and $151 million which has not yet been billed nor recognized as revenue. Approximately 39% of revenue from contracts greater than one year in length will be recognized over the next twelve months.'}}"
      ]
     },
     "execution_count": null,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "nvda_10k_extract.data"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "20f643ec",
   "metadata": {},
   "source": [
    "#### Verifying Financial Metrics\n",
    "\n",
    "Now let use the page numbers to verify the accuracy of the financial metrics extracted.\n",
    "\n",
    "Here's the relevant financial metrics extracted:\n",
    "\n",
    "```python\n",
    "{\n",
    " 'financial_highlights': {'period_end': '2025-01-26',\n",
    "  'comparison_period_end': '2024-01-28',\n",
    "  'currency': 'USD',\n",
    "  'unit': 'thousands',\n",
    "  'revenue': 130497.0,\n",
    "  'revenue_prior_period': 60922.0,\n",
    "  'revenue_growth': 114.23,\n",
    "  'gross_profit': 97858.0,\n",
    "  'gross_margin': 75.0,\n",
    "  'operating_income': 81453.0,\n",
    "  'operating_margin': None,\n",
    "  'net_income': 72880.0,\n",
    "  'net_margin': 55.8,\n",
    "  'eps': None,\n",
    "  'diluted_eps': None,\n",
    "  'ebitda': None,\n",
    "  'free_cash_flow': None,\n",
    "  'page_numbers': [40, 41, 55, 56, 68]},\n",
    "}\n",
    "```\n",
    "We can see that the gross margin of 75% is extracted fro page 40. The revenue number of 130,497 is extracted from page 41 which also has the breakdown of the revenue by segment.\n",
    "\n",
    "**Page 40 (showing gross margin of 75%):**\n",
    "<img src=\"./data/sec_filings/nvda_10k_page_40.png\" width=\"50%\" alt=\"NVIDIA 10K Page 40\">\n",
    "\n",
    "**Page 41 (showing revenue of 130,497):**\n",
    "<img src=\"./data/sec_filings/nvda_10k_page_41.png\" width=\"50%\" alt=\"NVIDIA 10K Page 41\">\n",
    "\n",
    "You can likewise verify that the geographic breakdown of revenue is extracted from page 79 correctly. "
   ]
  },
  {
   "cell_type": "markdown",
   "id": "af810951",
   "metadata": {},
   "source": [
    "### General Guidelines for Long-Form Document Extraction\n",
    "\n",
    "- **Schema Iteration using the Web UI**: We have a Web UI with a schema builder that can help you define your schema and iterate on different documents. We have a 10-K/Q schema for you to get started with if you are interested in trying this out. \n",
    "  Start small and build from there! Refer to the tips above. Try your schema on different documents to see whether it generalizes to the target documents.\n",
    "- **Citations**: You can ask the extraction agent to provide page numbers for key figures extracted. This will help you quickly navigate to the relevant section of the document and verify the veracity of the information extracted. \n",
    "  We will have a more robust and convenient citation feature in the future. \n",
    "- **Run scalable batch jobs**: Once you have confidence that the extraction agent is working well, you can use your agent via our [Python SDK](https://github.com/run-llama/llama_cloud_services) to run scalable batch jobs. \n",
    "\n",
    "![Web UI with the 10-K/Q Template](./data/sec_filings/web_ui.png)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "llama-cloud-services",
   "language": "python",
   "name": "llama-cloud-services"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "f2a4d1d8",
   "metadata": {},
   "source": [
    "# Automotive Equity Research: A Multi-Step Agentic Workflow\n",
    "\n",
    "<a href=\"https://colab.research.google.com/github/run-llama/llama_cloud_services/blob/main/examples/extract/automotive_sector_analysis.ipynb\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>\n",
    "\n",
    "This notebook demonstrates an end‑to‑end agentic workflow using LlamaExtract and the LlamaIndex event‑driven workflow framework for automotive sector analysis.\n",
    "\n",
    "In this workflow, we:\n",
    "1. **Extract** key financial metrics from Q2 2024 earnings reports for Tesla and Ford.\n",
    "2. **Generate** a preliminary financial model summary for each company using an LLM.\n",
    "3. **Cross‑reference** Tesla's metrics with Ford's data to produce a final equity research memo.\n",
    "4. **Output** the memo as structured JSON.\n",
    "\n",
    "This workflow is designed for equity research analysts and investment professionals."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e7979faf",
   "metadata": {},
   "source": [
    "> **⚠️ DEPRECATION NOTICE**>> This example uses the deprecated `llama-cloud-services` package, which will be maintained until **May 1, 2026**.>> **Please migrate to:**> - **Python**: `pip install llama-cloud>=1.0` ([GitHub](https://github.com/run-llama/llama-cloud-py))> - **New Package Documentation**: https://docs.cloud.llamaindex.ai/>> The new package provides the same functionality with improved performance and support."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "880ea07c",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Download an example earnings call transcript PDF from SEC EDGAR (Tesla Q2 Earnings as an example)\n",
    "!mkdir -p data/automotive_sector_analysis\n",
    "!wget https://digitalassets.tesla.com/tesla-contents/image/upload/IR/TSLA-Q2-2024-Update.pdf -O data/automotive_sector_analysis/tesla_q2_earnings.pdf\n",
    "!wget https://s205.q4cdn.com/882619693/files/doc_financials/2024/q2/Q2-2024-Ford-Earnings-Press-Release.pdf -O data/automotive_sector_analysis/ford_q2_earnings_press_release.pdf"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "4f2b2ea4",
   "metadata": {},
   "source": [
    "## Define the Output Schema\n",
    "\n",
    "We define a schema to represent the final equity research memo. This includes the company name, a summary of the financial model, a comparative analysis, and an overall recommendation."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "492f8bf5",
   "metadata": {},
   "outputs": [],
   "source": [
    "from pydantic import BaseModel, Field\n",
    "from typing import Optional\n",
    "\n",
    "\n",
    "class RawFinancials(BaseModel):\n",
    "    revenue: Optional[float] = Field(\n",
    "        None, description=\"Extracted revenue (in million USD)\"\n",
    "    )\n",
    "    operating_income: Optional[float] = Field(\n",
    "        None, description=\"Extracted operating income (in million USD)\"\n",
    "    )\n",
    "    eps: Optional[float] = Field(None, description=\"Extracted earnings per share\")\n",
    "    # Add more metrics as needed\n",
    "\n",
    "\n",
    "class InitialFinancialDataOutput(BaseModel):\n",
    "    company_name: str = Field(\n",
    "        ..., description=\"Company name as extracted from the earnings deck\"\n",
    "    )\n",
    "    ticker: str = Field(..., description=\"Stock ticker symbol\")\n",
    "    report_date: str = Field(..., description=\"Date of the earnings deck/report\")\n",
    "    raw_financials: RawFinancials = Field(\n",
    "        ..., description=\"Structured raw financial metrics\"\n",
    "    )\n",
    "    narrative: Optional[str] = Field(\n",
    "        None, description=\"Additional narrative content (if any)\"\n",
    "    )"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "1441e681",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "FinalEquityResearchMemoOutput schema defined.\n"
     ]
    }
   ],
   "source": [
    "from pydantic import BaseModel, Field\n",
    "from typing import List, Dict\n",
    "\n",
    "\n",
    "# Define the structured output schema for each company's financial model\n",
    "class FinancialModelOutput(BaseModel):\n",
    "    revenue_projection: float = Field(\n",
    "        ..., description=\"Projected revenue for next year (in million USD)\"\n",
    "    )\n",
    "    operating_income_projection: float = Field(\n",
    "        ..., description=\"Projected operating income for next year (in million USD)\"\n",
    "    )\n",
    "    growth_rate: float = Field(..., description=\"Expected revenue growth rate (%)\")\n",
    "    discount_rate: float = Field(\n",
    "        ..., description=\"Discount rate (%) used for valuation\"\n",
    "    )\n",
    "    terminal_growth_rate: float = Field(\n",
    "        ..., description=\"Terminal growth rate (%) used in the model\"\n",
    "    )\n",
    "    valuation_estimate: float = Field(\n",
    "        ..., description=\"Estimated enterprise value (in million USD)\"\n",
    "    )\n",
    "    key_assumptions: str = Field(\n",
    "        ..., description=\"Key assumptions such as tax rate, CAPEX ratio, etc.\"\n",
    "    )\n",
    "    summary: str = Field(\n",
    "        ..., description=\"A brief summary of the preliminary financial model analysis.\"\n",
    "    )\n",
    "\n",
    "\n",
    "class ComparativeAnalysisOutput(BaseModel):\n",
    "    comparative_analysis: str = Field(\n",
    "        ..., description=\"Comparative analysis between Company A and Company B\"\n",
    "    )\n",
    "    overall_recommendation: str = Field(\n",
    "        ..., description=\"Overall investment recommendation with rationale\"\n",
    "    )\n",
    "\n",
    "\n",
    "# Define the final equity research memo schema, which aggregates the outputs for Company A and B\n",
    "class FinalEquityResearchMemoOutput(BaseModel):\n",
    "    company_a_model: FinancialModelOutput = Field(\n",
    "        ..., description=\"Financial model summary for Company A\"\n",
    "    )\n",
    "    company_b_model: FinancialModelOutput = Field(\n",
    "        ..., description=\"Financial model summary for Company B\"\n",
    "    )\n",
    "    comparative_analysis: ComparativeAnalysisOutput = Field(\n",
    "        ..., description=\"Comparative analysis between Company A and Company B\"\n",
    "    )\n",
    "\n",
    "\n",
    "print(\"FinalEquityResearchMemoOutput schema defined.\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "c3f8bd67",
   "metadata": {},
   "source": [
    "## Initialize the Extraction Agent\n",
    "\n",
    "We create (or replace) an extraction agent using our automotive sector analysis schema."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "50ea51d7",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Automotive sector analysis extraction agent created.\n"
     ]
    }
   ],
   "source": [
    "from dotenv import load_dotenv\n",
    "from llama_cloud_services import LlamaExtract\n",
    "from llama_cloud.core.api_error import ApiError\n",
    "from llama_cloud import ExtractConfig\n",
    "\n",
    "\n",
    "llama_extract = LlamaExtract(\n",
    "    project_id=\"2fef999e-1073-40e6-aeb3-1f3c0e64d99b\",\n",
    "    organization_id=\"43b88c8f-e488-46f6-9013-698e3d2e374a\",\n",
    ")\n",
    "\n",
    "try:\n",
    "    existing_agent = llama_extract.get_agent(name=\"automotive-sector-analysis\")\n",
    "    if existing_agent:\n",
    "        llama_extract.delete_agent(existing_agent.id)\n",
    "except ApiError as e:\n",
    "    if e.status_code == 404:\n",
    "        pass\n",
    "    else:\n",
    "        raise\n",
    "\n",
    "extract_config = ExtractConfig(\n",
    "    extraction_mode=\"BALANCED\"\n",
    "    # extraction_mode=\"MULTIMODAL\"\n",
    ")\n",
    "\n",
    "agent = llama_extract.create_agent(\n",
    "    name=\"automotive-sector-analysis\",\n",
    "    data_schema=InitialFinancialDataOutput,\n",
    "    config=extract_config,\n",
    ")\n",
    "print(\"Automotive sector analysis extraction agent created.\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "ecf5cce2",
   "metadata": {},
   "source": [
    "## Define the Workflow\n",
    "\n",
    "This workflow analyzes Q2 2024 earnings reports for two major automotive companies:\n",
    "\n",
    "- **Tesla (TSLA)**: Focus on electric vehicles, energy storage, and regulatory credits\n",
    "- **Ford (F)**: Traditional automotive manufacturer with growing EV segment\n",
    "\n",
    "Key metrics extracted and analyzed:\n",
    "- Revenue and revenue projections\n",
    "- Operating income\n",
    "- Growth rates\n",
    "- Valuation estimates\n",
    "- Key business segment performance\n",
    "\n",
    "In this workflow, the steps are:\n",
    "1. **parse_transcript:** Extract text (with page citations) from the earnings call transcript PDF.\n",
    "2. **load_modeling_data:** Load financial modeling assumptions from a text file.\n",
    "3. **generate_financial_model:** Generate a preliminary financial model summary using an LLM.\n",
    "4. **load_comparable_data:** **Extract** comparable financial metrics from a PDF file (Company B).\n",
    "5. **cross_reference:** Compare Company A’s metrics with Company B’s data using the LLM.\n",
    "6. **output_final_memo:** Assemble the final equity research memo and output it as JSON."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "c1f8b2f4",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/Users/jerryliu/Programming/llama_parse/.venv/lib/python3.10/site-packages/pydantic/_internal/_fields.py:132: UserWarning: Field \"model_output\" in CompanyModelEvent has conflict with protected namespace \"model_\".\n",
      "\n",
      "You may be able to resolve this warning by setting `model_config['protected_namespaces'] = ()`.\n",
      "  warnings.warn(\n"
     ]
    }
   ],
   "source": [
    "from llama_index.core.workflow import (\n",
    "    Event,\n",
    "    StartEvent,\n",
    "    StopEvent,\n",
    "    Context,\n",
    "    Workflow,\n",
    "    step,\n",
    ")\n",
    "from llama_index.llms.openai import OpenAI\n",
    "from llama_index.core.llms.llm import LLM\n",
    "from llama_index.core.prompts import ChatPromptTemplate\n",
    "\n",
    "\n",
    "# Define custom events for each step\n",
    "class DeckAParseEvent(Event):\n",
    "    deck_content: InitialFinancialDataOutput\n",
    "\n",
    "\n",
    "class DeckBParseEvent(Event):\n",
    "    deck_content: InitialFinancialDataOutput\n",
    "\n",
    "\n",
    "class CompanyModelEvent(Event):\n",
    "    model_output: FinancialModelOutput\n",
    "\n",
    "\n",
    "class ComparableDataLoadEvent(Event):\n",
    "    company_a_output: FinancialModelOutput\n",
    "    company_b_output: FinancialModelOutput\n",
    "\n",
    "\n",
    "class LogEvent(Event):\n",
    "    msg: str\n",
    "    delta: bool = False\n",
    "\n",
    "\n",
    "class AutomotiveSectorAnalysisWorkflow(Workflow):\n",
    "    \"\"\"\n",
    "    Workflow to generate an equity research memo for automotive sector analysis.\n",
    "    \"\"\"\n",
    "\n",
    "    def __init__(\n",
    "        self,\n",
    "        agent: LlamaExtract,\n",
    "        modeling_path: str,\n",
    "        llm: Optional[LLM] = None,\n",
    "        **kwargs\n",
    "    ):\n",
    "        super().__init__(**kwargs)\n",
    "        self.agent = agent\n",
    "        self.llm = llm or OpenAI(model=\"o3-mini\")\n",
    "        # Load financial modeling assumptions from file\n",
    "        with open(modeling_path, \"r\") as f:\n",
    "            self.modeling_data = f.read()\n",
    "        # Instead of loading comparable data from a text file, we load from a PDF\n",
    "\n",
    "    async def _parse_deck(self, ctx: Context, deck_path) -> InitialFinancialDataOutput:\n",
    "        extraction_result = await self.agent.aextract(deck_path)\n",
    "        initial_output = extraction_result.data  # expected to be a string\n",
    "        ctx.write_event_to_stream(LogEvent(msg=\"Transcript parsed successfully.\"))\n",
    "        return initial_output\n",
    "\n",
    "    @step\n",
    "    async def parse_deck_a(self, ctx: Context, ev: StartEvent) -> DeckAParseEvent:\n",
    "        initial_output = await self._parse_deck(ctx, ev.deck_path_a)\n",
    "        await ctx.set(\"initial_output_a\", initial_output)\n",
    "        return DeckAParseEvent(deck_content=initial_output)\n",
    "\n",
    "    @step\n",
    "    async def parse_deck_b(self, ctx: Context, ev: StartEvent) -> DeckBParseEvent:\n",
    "        initial_output = await self._parse_deck(ctx, ev.deck_path_b)\n",
    "        await ctx.set(\"initial_output_b\", initial_output)\n",
    "        return DeckBParseEvent(deck_content=initial_output)\n",
    "\n",
    "    async def _generate_financial_model(\n",
    "        self, ctx: Context, financial_data: InitialFinancialDataOutput\n",
    "    ) -> FinancialModelOutput:\n",
    "        prompt_str = \"\"\"\n",
    "    You are an expert financial analyst.\n",
    "    Using the following raw financial data from an earnings deck and financial modeling assumptions,\n",
    "    refine the data to produce a financial model summary. Adjust the assumptions based on the company-specific context.\n",
    "    Please use the most recent quarter's financial data from the earnings deck.\n",
    "\n",
    "    Raw Financial Data:\n",
    "    {raw_data}\n",
    "    Financial Modeling Assumptions:\n",
    "    {assumptions}\n",
    "\n",
    "    Return your output as JSON conforming to the FinancialModelOutput schema.\n",
    "    You MUST make sure all fields are filled in the output JSON.\n",
    "\n",
    "    \"\"\"\n",
    "        prompt = ChatPromptTemplate.from_messages([(\"user\", prompt_str)])\n",
    "        refined_model = await self.llm.astructured_predict(\n",
    "            FinancialModelOutput,\n",
    "            prompt,\n",
    "            raw_data=financial_data.model_dump_json(),\n",
    "            assumptions=self.modeling_data,\n",
    "        )\n",
    "        return refined_model\n",
    "\n",
    "    @step\n",
    "    async def refine_financial_model_company_a(\n",
    "        self, ctx: Context, ev: DeckAParseEvent\n",
    "    ) -> CompanyModelEvent:\n",
    "        print(\"deck content A\", ev.deck_content)\n",
    "        refined_model = await self._generate_financial_model(ctx, ev.deck_content)\n",
    "        print(\"refined_model A\", refined_model)\n",
    "        print(type(refined_model))\n",
    "        await ctx.set(\"CompanyAModelEvent\", refined_model)\n",
    "        return CompanyModelEvent(model_output=refined_model)\n",
    "\n",
    "    @step\n",
    "    async def refine_financial_model_company_b(\n",
    "        self, ctx: Context, ev: DeckBParseEvent\n",
    "    ) -> CompanyModelEvent:\n",
    "        print(\"deck content B\", ev.deck_content)\n",
    "        refined_model = await self._generate_financial_model(ctx, ev.deck_content)\n",
    "        print(\"refined_model B\", refined_model)\n",
    "        print(type(refined_model))\n",
    "        await ctx.set(\"CompanyBModelEvent\", refined_model)\n",
    "        return CompanyModelEvent(model_output=refined_model)\n",
    "\n",
    "    @step\n",
    "    async def cross_reference_models(\n",
    "        self, ctx: Context, ev: CompanyModelEvent\n",
    "    ) -> StopEvent:\n",
    "        # Assume CompanyAModelEvent and CompanyBModelEvent are stored in the context\n",
    "        company_a_model = await ctx.get(\"CompanyAModelEvent\", default=None)\n",
    "        company_b_model = await ctx.get(\"CompanyBModelEvent\", default=None)\n",
    "        if company_a_model is None or company_b_model is None:\n",
    "            return\n",
    "\n",
    "        prompt_str = \"\"\"\n",
    "    You are an expert investment analyst.\n",
    "    Compare the following refined financial models for Company A and Company B.\n",
    "    Based on this comparison, provide a specific investment recommendation for Tesla (Company A).\n",
    "    Focus your analysis on:\n",
    "    1. Key differences in revenue projections, operating income, and growth rates\n",
    "    2. Valuation estimates and their implications\n",
    "    3. Clear recommendation for Tesla with supporting rationale\n",
    "    Return your analysis as plain text.\n",
    "    Company A Model:\n",
    "    {company_a_model}\n",
    "    Company B Model:\n",
    "    {company_b_model}\n",
    "    \"\"\"\n",
    "        prompt = ChatPromptTemplate.from_messages([(\"user\", prompt_str)])\n",
    "        comp_analysis = await self.llm.astructured_predict(\n",
    "            ComparativeAnalysisOutput,\n",
    "            prompt,\n",
    "            company_a_model=company_a_model.model_dump_json(),\n",
    "            company_b_model=company_b_model.model_dump_json(),\n",
    "        )\n",
    "        final_memo = FinalEquityResearchMemoOutput(\n",
    "            company_a_model=company_a_model,\n",
    "            company_b_model=company_b_model,\n",
    "            comparative_analysis=comp_analysis,\n",
    "        )\n",
    "        return StopEvent(result={\"memo\": final_memo})"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "a8a3a8c1",
   "metadata": {},
   "source": [
    "## Running the Workflow\n",
    "\n",
    "Now we run the workflow with the pre-loaded modeling assumptions and the deck from both companies."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "a4d5f42c",
   "metadata": {},
   "outputs": [],
   "source": [
    "import nest_asyncio\n",
    "\n",
    "nest_asyncio.apply()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "be767dc9",
   "metadata": {},
   "outputs": [],
   "source": [
    "modeling_path = \"./data/automotive_sector_analysis/modeling_assumptions.txt\"\n",
    "workflow = AutomotiveSectorAnalysisWorkflow(\n",
    "    agent=agent, modeling_path=modeling_path, verbose=True, timeout=240\n",
    ")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "cf68b0c7",
   "metadata": {},
   "source": [
    "#### Visualize the Workflow\n",
    "\n",
    "![](data/automotive_sector_analysis/workflow_img.png)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "5b90a486",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'NoneType'>\n",
      "<class 'llama_index.core.workflow.events.StopEvent'>\n",
      "<class '__main__.DeckAParseEvent'>\n",
      "<class '__main__.DeckBParseEvent'>\n",
      "<class '__main__.CompanyModelEvent'>\n",
      "<class '__main__.CompanyModelEvent'>\n",
      "automotive_sector_analysis_workflow.html\n"
     ]
    }
   ],
   "source": [
    "from llama_index.utils.workflow import draw_all_possible_flows\n",
    "\n",
    "draw_all_possible_flows(\n",
    "    AutomotiveSectorAnalysisWorkflow,\n",
    "    filename=\"automotive_sector_analysis_workflow.html\",\n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "70493e4f",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Running step parse_deck_a\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Uploading files:   0%|          | 0/1 [00:00<?, ?it/s]"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Running step parse_deck_b\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": []
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Uploading files: 100%|██████████| 1/1 [00:00<00:00,  1.13it/s]\n",
      "Creating extraction jobs: 100%|██████████| 1/1 [00:00<00:00,  3.87it/s]\n",
      "Uploading files: 100%|██████████| 1/1 [00:01<00:00,  1.25s/it]\n",
      "Creating extraction jobs: 100%|██████████| 1/1 [00:00<00:00,  4.05it/s]\n",
      "Extracting files: 100%|██████████| 1/1 [00:03<00:00,  3.82s/it]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Step parse_deck_b produced event DeckBParseEvent\n",
      "Running step refine_financial_model_company_b\n",
      "deck content B company_name='Ford Motor Company' ticker='F' report_date='July 24, 2024' raw_financials=RawFinancials(revenue=47.8, operating_income=2.8, eps=0.46) narrative='Ford reports second-quarter revenue of $47.8 billion, net income of $1.8 billion and adjusted EBIT of $2.8 billion. Ford Pro posts quarterly EBIT of $2.6 billion – a 15% margin – on 9% revenue gain; customers buying every Super Duty truck and Transit van the company can make. Ford Blue hybrid sales up 34%, represent nearly 9% of company’s global vehicle mix; Ford Model e costs down ~$400 million. Expectations for full-year 2024 adjusted EBIT unchanged at $10 billion to $12 billion; adjusted free cash flow outlook raised $1 billion, to between $7.5 billion and $8.5 billion.'\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Extracting files: 100%|██████████| 1/1 [00:03<00:00,  3.74s/it]\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Step parse_deck_a produced event DeckAParseEvent\n",
      "Running step refine_financial_model_company_a\n",
      "deck content A company_name='Tesla' ticker='TSLA' report_date='Q2 2024' raw_financials=RawFinancials(revenue=25.5, operating_income=1.6, eps=0.42) narrative='In Q2, we achieved record quarterly revenues despite a difficult operating environment. The Energy Storage business continues to grow rapidly, setting a record in Q2 with 9.4 GWh of deployments, resulting in record revenues and gross profits for the overall segment. We also saw a sequential rebound in vehicle deliveries in Q2 as overall consumer sentiment improved and we launched attractive financing options to offset the impact of sustained high interest rates. We recognized record regulatory credit revenues in Q2 as other OEMs are still behind on meeting emissions requirements. Global EV penetration returned to growth in Q2 and is taking share from ICE vehicles. We believe that a pure EV is the optimal vehicle design and will ultimately win over consumers as the myths on range, charging and service are debunked.'\n",
      "refined_model A revenue_projection=112.2 operating_income_projection=7.03 growth_rate=10.0 discount_rate=8.0 terminal_growth_rate=2.0 valuation_estimate=87.93 key_assumptions='Discount Rate: 8%, Terminal Growth Rate: 2%, Tax Rate: 25%, Revenue Growth (Years 1-5): 10% per annum, Revenue Growth (Years 6-10): 5% per annum, CAPEX as 7% of Revenue, Working Capital at 3% of Revenue, Depreciation at 10% per annum, Cost of Capital: 8%.' summary=\"Based on Tesla's Q2 2024 earnings where record quarterly revenue of $25.5 million (quarterly) was achieved, we annualized the performance (multiplied by 4) to establish a baseline annual revenue of approximately $102 million. Applying a conservative growth rate of 10% for next year leads to a revenue projection of around $112.2 million. Maintaining operating margin derived from Q2 (approximately 6.27%), the projected operating income is about $7.03 million. Utilizing a simplified DCF approach – adding back non-cash depreciation, subtracting CAPEX and working capital needs – the estimated enterprise value comes to roughly $87.93 million, using the given discount rate of 8% and terminal growth rate of 2%. These projections reflect Tesla’s strong performance in energy storage and vehicle deliveries amid challenging operating conditions, underpinned by robust demand drivers such as regulatory credits and rapid EV market expansion.\"\n",
      "<class '__main__.FinancialModelOutput'>\n",
      "Step refine_financial_model_company_a produced event CompanyModelEvent\n",
      "Running step cross_reference_models\n",
      "Step cross_reference_models produced no event\n",
      "refined_model B revenue_projection=210320.0 operating_income_projection=12320.0 growth_rate=10.0 discount_rate=8.0 terminal_growth_rate=2.0 valuation_estimate=154000.0 key_assumptions='Revenue growth of 10% per annum is applied for the next year, based on replicating Q2 values on an annual basis (by multiplying quarterly revenue by 4). Operating income is assumed to grow in line with revenue, maintaining the current operating margin. The tax rate is set at 25%, with capital expenditures estimated at 7% of revenue, working capital requirements at 3% of revenue, and depreciation at 10% of revenue. An 8% discount rate and a terminal growth rate of 2% are used to compute a simplified perpetual growth valuation using free cash flow derived from after-tax operating income adjusted for non-cash depreciation and reinvestment needs.' summary=\"Using Ford Motor Company's most recent Q2 financial data, we annualized the revenue (47.8 billion USD for Q2 becomes approximately 191.2 billion USD on an annual basis) and operating income figures to form our baseline. With an assumed annual revenue growth of 10%, next year's projected revenue is about 210.32 billion USD and operating income about 12.32 billion USD. Adjustments for taxes, CAPEX, working capital, and depreciation suggest a free cash flow of roughly 9.24 billion USD. Applying an 8% discount rate and a terminal growth rate of 2% yields an estimated enterprise value of approximately 154 billion USD. The model reflects a consistent margin profile and incorporates the company-specific outlook provided in the earnings deck.\"\n",
      "<class '__main__.FinancialModelOutput'>\n",
      "Step refine_financial_model_company_b produced event CompanyModelEvent\n",
      "Running step cross_reference_models\n",
      "Step cross_reference_models produced event StopEvent\n",
      "\n",
      "********Final Equity Research Memo:********\n",
      " company_a_model=FinancialModelOutput(revenue_projection=112.2, operating_income_projection=7.03, growth_rate=10.0, discount_rate=8.0, terminal_growth_rate=2.0, valuation_estimate=87.93, key_assumptions='Discount Rate: 8%, Terminal Growth Rate: 2%, Tax Rate: 25%, Revenue Growth (Years 1-5): 10% per annum, Revenue Growth (Years 6-10): 5% per annum, CAPEX as 7% of Revenue, Working Capital at 3% of Revenue, Depreciation at 10% per annum, Cost of Capital: 8%.', summary=\"Based on Tesla's Q2 2024 earnings where record quarterly revenue of $25.5 million (quarterly) was achieved, we annualized the performance (multiplied by 4) to establish a baseline annual revenue of approximately $102 million. Applying a conservative growth rate of 10% for next year leads to a revenue projection of around $112.2 million. Maintaining operating margin derived from Q2 (approximately 6.27%), the projected operating income is about $7.03 million. Utilizing a simplified DCF approach – adding back non-cash depreciation, subtracting CAPEX and working capital needs – the estimated enterprise value comes to roughly $87.93 million, using the given discount rate of 8% and terminal growth rate of 2%. These projections reflect Tesla’s strong performance in energy storage and vehicle deliveries amid challenging operating conditions, underpinned by robust demand drivers such as regulatory credits and rapid EV market expansion.\") company_b_model=FinancialModelOutput(revenue_projection=210320.0, operating_income_projection=12320.0, growth_rate=10.0, discount_rate=8.0, terminal_growth_rate=2.0, valuation_estimate=154000.0, key_assumptions='Revenue growth of 10% per annum is applied for the next year, based on replicating Q2 values on an annual basis (by multiplying quarterly revenue by 4). Operating income is assumed to grow in line with revenue, maintaining the current operating margin. The tax rate is set at 25%, with capital expenditures estimated at 7% of revenue, working capital requirements at 3% of revenue, and depreciation at 10% of revenue. An 8% discount rate and a terminal growth rate of 2% are used to compute a simplified perpetual growth valuation using free cash flow derived from after-tax operating income adjusted for non-cash depreciation and reinvestment needs.', summary=\"Using Ford Motor Company's most recent Q2 financial data, we annualized the revenue (47.8 billion USD for Q2 becomes approximately 191.2 billion USD on an annual basis) and operating income figures to form our baseline. With an assumed annual revenue growth of 10%, next year's projected revenue is about 210.32 billion USD and operating income about 12.32 billion USD. Adjustments for taxes, CAPEX, working capital, and depreciation suggest a free cash flow of roughly 9.24 billion USD. Applying an 8% discount rate and a terminal growth rate of 2% yields an estimated enterprise value of approximately 154 billion USD. The model reflects a consistent margin profile and incorporates the company-specific outlook provided in the earnings deck.\") comparative_analysis=ComparativeAnalysisOutput(comparative_analysis='Comparing the two refined models reveals several key differences. Company A’s (Tesla’s) model is based on a much smaller revenue base (annualized quarterly revenue of approximately $102 million, growing to $112.2 million with a 10% growth assumption) compared to Company B’s (Ford’s) model, which scales from a considerably larger quarterly figure to a projected annual revenue of roughly $210.32 billion. In Tesla’s model, the operating income projection is about $7.03 million, derived from maintaining a consistent operating margin from a strong Q2 performance. In contrast, Ford’s operating income is projected at about $12.32 billion, reflecting its vast operational scale and established footprint. \\n\\nBoth models assume similar growth rates (10% next year) and employ the same discount (8%) and terminal growth (2%) rates, but the underlying business model assumptions (e.g., CAPEX, working capital, and depreciation percentages) are applied to very different revenue levels, underscoring the distinct corporate scales and market positions. Additionally, the valuation estimate for Tesla is markedly lower at approximately $87.93 million compared to Ford’s multi-billion-dollar enterprise value, which highlights the relative stage and market capitalization disparities between the high-growth, innovation-driven Tesla and the mature, large-scale operation of Ford. \\n\\nFurthermore, Tesla’s model emphasizes its momentum in sectors like energy storage and EV deliveries, driven by factors such as regulatory credits and rapid market expansion, whereas Ford’s model reflects a traditional automotive giant with steady, albeit massive, revenue growth. \\n\\nOverall, the analysis—despite both companies using similar underlying financial assumptions—demonstrates that Tesla’s smaller scale and agility allow it to potentially achieve higher growth rates and innovative disruption, even though the absolute revenue and operating income values are much lower than Ford’s. This distinction is critical in the investment decision process.', overall_recommendation='Based on the comparative analysis, I recommend an investment in Tesla (Company A). Tesla’s refined model reflects robust growth potential, driven by its strong Q2 performance, efficient operating margins, and strategic positioning in the high-growth EV and energy storage markets. Despite its smaller scale compared to Ford, Tesla’s ability to capitalize on emerging demand drivers, innovation in technology, and future market expansion serves as a compelling case for investment. This recommendation rests on the expectation that Tesla’s growth trajectory and market disruption potential will enable it to outperform traditional automotive peers in a rapidly evolving industry, making it a strategic buy for investors seeking exposure to transformative technological and market trends.')\n"
     ]
    }
   ],
   "source": [
    "result = await workflow.run(\n",
    "    deck_path_a=\"./data/automotive_sector_analysis/tesla_q2_earnings.pdf\",\n",
    "    deck_path_b=\"./data/automotive_sector_analysis/ford_q2_earnings_press_release.pdf\",\n",
    ")\n",
    "final_memo = result[\"memo\"]\n",
    "print(\"\\n********Final Equity Research Memo:********\\n\", final_memo)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "f00d58e0",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "ComparativeAnalysisOutput(comparative_analysis='Comparing the two refined models reveals several key differences. Company A’s (Tesla’s) model is based on a much smaller revenue base (annualized quarterly revenue of approximately $102 million, growing to $112.2 million with a 10% growth assumption) compared to Company B’s (Ford’s) model, which scales from a considerably larger quarterly figure to a projected annual revenue of roughly $210.32 billion. In Tesla’s model, the operating income projection is about $7.03 million, derived from maintaining a consistent operating margin from a strong Q2 performance. In contrast, Ford’s operating income is projected at about $12.32 billion, reflecting its vast operational scale and established footprint. \\n\\nBoth models assume similar growth rates (10% next year) and employ the same discount (8%) and terminal growth (2%) rates, but the underlying business model assumptions (e.g., CAPEX, working capital, and depreciation percentages) are applied to very different revenue levels, underscoring the distinct corporate scales and market positions. Additionally, the valuation estimate for Tesla is markedly lower at approximately $87.93 million compared to Ford’s multi-billion-dollar enterprise value, which highlights the relative stage and market capitalization disparities between the high-growth, innovation-driven Tesla and the mature, large-scale operation of Ford. \\n\\nFurthermore, Tesla’s model emphasizes its momentum in sectors like energy storage and EV deliveries, driven by factors such as regulatory credits and rapid market expansion, whereas Ford’s model reflects a traditional automotive giant with steady, albeit massive, revenue growth. \\n\\nOverall, the analysis—despite both companies using similar underlying financial assumptions—demonstrates that Tesla’s smaller scale and agility allow it to potentially achieve higher growth rates and innovative disruption, even though the absolute revenue and operating income values are much lower than Ford’s. This distinction is critical in the investment decision process.', overall_recommendation='Based on the comparative analysis, I recommend an investment in Tesla (Company A). Tesla’s refined model reflects robust growth potential, driven by its strong Q2 performance, efficient operating margins, and strategic positioning in the high-growth EV and energy storage markets. Despite its smaller scale compared to Ford, Tesla’s ability to capitalize on emerging demand drivers, innovation in technology, and future market expansion serves as a compelling case for investment. This recommendation rests on the expectation that Tesla’s growth trajectory and market disruption potential will enable it to outperform traditional automotive peers in a rapidly evolving industry, making it a strategic buy for investors seeking exposure to transformative technological and market trends.')"
      ]
     },
     "execution_count": null,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "final_memo.comparative_analysis"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "llama_parse",
   "language": "python",
   "name": "llama_parse"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Extracting data from Resumes\n",
    "\n",
    "Let us assume that we are running a hiring process for a company and we have received a list of resumes from candidates. We want to extract structured data from the resumes so that we can run a screening process and shortlist candidates. \n",
    "\n",
    "Take a look at one of the resumes in the `data/resumes` directory. "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "> **⚠️ DEPRECATION NOTICE**>> This example uses the deprecated `llama-cloud-services` package, which will be maintained until **May 1, 2026**.>> **Please migrate to:**> - **Python**: `pip install llama-cloud>=1.0` ([GitHub](https://github.com/run-llama/llama-cloud-py))> - **New Package Documentation**: https://docs.cloud.llamaindex.ai/>> The new package provides the same functionality with improved performance and support."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "\n",
       "        <iframe\n",
       "            width=\"600\"\n",
       "            height=\"400\"\n",
       "            src=\"./data/resumes/ai_researcher.pdf\"\n",
       "            frameborder=\"0\"\n",
       "            allowfullscreen\n",
       "            \n",
       "        ></iframe>\n",
       "        "
      ],
      "text/plain": [
       "<IPython.lib.display.IFrame at 0x109a7dcd0>"
      ]
     },
     "execution_count": null,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from IPython.display import IFrame\n",
    "\n",
    "IFrame(src=\"./data/resumes/ai_researcher.pdf\", width=600, height=400)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "You will notice that all the resumes have different layouts but contain common information like name, email, experience, education, etc. \n",
    "\n",
    "With LlamaExtract, we will show you how to:\n",
    "- *Define* a data schema to extract the information of interest. \n",
    "- *Iterate* over the data schema to generalize the schema for multiple resumes.\n",
    "- *Finalize* the schema and schedule extractions for multiple resumes.\n",
    "\n",
    "We will start by defining a `LlamaExtract` client which provides a Python interface to the LlamaExtract API. "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from dotenv import load_dotenv\n",
    "from llama_cloud_services import LlamaExtract\n",
    "\n",
    "\n",
    "# Load environment variables (put LLAMA_CLOUD_API_KEY in your .env file)\n",
    "load_dotenv(override=True)\n",
    "\n",
    "# Optionally, add your project id/organization id\n",
    "llama_extract = LlamaExtract()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Defining the data schema\n",
    "\n",
    "Next, let us try to extract two fields from the resume: `name` and `email`. We can either use a Python dictionary structure to define the `data_schema` as a JSON or use a Pydantic model instead, for brevity and convenience. In either case, our output is guaranteed to validate against this schema."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from pydantic import BaseModel, Field\n",
    "\n",
    "\n",
    "class Resume(BaseModel):\n",
    "    name: str = Field(description=\"The name of the candidate\")\n",
    "    email: str = Field(description=\"The email address of the candidate\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Uploading files: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:02<00:00,  2.20s/it]\n",
     ]
    }
   ],
   "source": [
    "from llama_cloud.core.api_error import ApiError\n",
    "\n",
    "try:\n",
    "    existing_agent = llama_extract.get_agent(name=\"resume-screening\")\n",
    "    if existing_agent:\n",
    "        llama_extract.delete_agent(existing_agent.id)\n",
    "except ApiError as e:\n",
    "    if e.status_code == 404:\n",
    "        pass\n",
    "    else:\n",
    "        raise\n",
    "\n",
    "agent = llama_extract.create_agent(name=\"resume-screening\", data_schema=Resume)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[ExtractionAgent(id=1fef43b5-8230-43b4-9e80-c1cddf53889c, name=resume-screening),\n",
       " ExtractionAgent(id=93f8508b-3570-46f0-ae62-6315b40043bd, name=receipt/noisebridge_receipt.pdf_56db3d92),\n",
       " ExtractionAgent(id=08315f0e-7146-430b-99b8-9701cb3ace6a, name=receipt/noisebridge_receipt.pdf_5c4730a7),\n",
       " ExtractionAgent(id=cfcd7756-015d-4dbd-b142-a3eefcb16cd3, name=resume/software_architect_resume.html_4a11cf15),\n",
       " ExtractionAgent(id=17cb83d9-601e-4f5c-a7aa-286e3045bcb4, name=resume/software_architect_resume.html_0b7d84a8),\n",
       " ExtractionAgent(id=adc8e88c-44d3-4613-a5aa-d666ef007494, name=slide/saas_slide.pdf_bcc627a5),\n",
       " ExtractionAgent(id=189f14cd-6370-4476-a6ad-36eafbc62618, name=slide/saas_slide.pdf_065aa22b),\n",
       " ExtractionAgent(id=b9938ca5-6225-43cb-89ea-b0065237792f, name=test2),\n",
       " ExtractionAgent(id=574d37b8-59dc-41e9-bde0-5c506a8eb670, name=test)]"
      ]
     },
     "execution_count": null,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "llama_extract.list_agents()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "{'name': 'Dr. Rachel Zhang', 'email': 'rachel.zhang@email.com'}"
      ]
     },
     "execution_count": null,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "resume = agent.extract(\"./data/resumes/ai_researcher.pdf\")\n",
    "resume.data"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Iterating over the data schema\n",
    "\n",
    "Now that we have created a data schema, let us add more fields to the schema. We will add `experience` and `education` fields to the schema. \n",
    "- We can create a new Pydantic model for each of these fields and represent `experience` and `education` as lists of these models. Doing this will allow us to extract multiple entities from the resume without having to pre-define how many experiences or education the candidate has. \n",
    "- We have added a `description` parameter to provide more context for extraction. We can use `description` to provide example inputs/outputs for the extraction. \n",
    "- Note that we have annotated the `start_date` and `end_date` fields with `Optional[str]` to indicate that these fields are optional. This is *important* because the schema will be used to extract data from multiple resumes and not all resumes will have the same format. A field must only be required if it is guaranteed to be present in all the resumes. \n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from typing import List, Optional\n",
    "\n",
    "\n",
    "class Education(BaseModel):\n",
    "    institution: str = Field(description=\"The institution of the candidate\")\n",
    "    degree: str = Field(description=\"The degree of the candidate\")\n",
    "    start_date: Optional[str] = Field(\n",
    "        default=None, description=\"The start date of the candidate's education\"\n",
    "    )\n",
    "    end_date: Optional[str] = Field(\n",
    "        default=None, description=\"The end date of the candidate's education\"\n",
    "    )\n",
    "\n",
    "\n",
    "class Experience(BaseModel):\n",
    "    company: str = Field(description=\"The name of the company\")\n",
    "    title: str = Field(description=\"The title of the candidate\")\n",
    "    description: Optional[str] = Field(\n",
    "        default=None, description=\"The description of the candidate's experience\"\n",
    "    )\n",
    "    start_date: Optional[str] = Field(\n",
    "        default=None, description=\"The start date of the candidate's experience\"\n",
    "    )\n",
    "    end_date: Optional[str] = Field(\n",
    "        default=None, description=\"The end date of the candidate's experience\"\n",
    "    )\n",
    "\n",
    "\n",
    "class Resume(BaseModel):\n",
    "    name: str = Field(description=\"The name of the candidate\")\n",
    "    email: str = Field(description=\"The email address of the candidate\")\n",
    "    links: List[str] = Field(\n",
    "        description=\"The links to the candidate's social media profiles\"\n",
    "    )\n",
    "    experience: List[Experience] = Field(description=\"The candidate's experience\")\n",
    "    education: List[Education] = Field(description=\"The candidate's education\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Next, we will update the `data_schema` for the `resume-screening` agent to use the new `Resume` model. "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "{'name': 'Dr. Rachel Zhang',\n",
       " 'email': 'rachel.zhang@email.com',\n",
       " 'links': ['linkedin.com/in/rachelzhang',\n",
       "  'github.com/rzhang-ai',\n",
       "  'scholar.google.com/rachelzhang'],\n",
       " 'experience': [{'company': 'DeepMind',\n",
       "   'title': 'Senior Research Scientist',\n",
       "   'description': '- Lead researcher on large-scale multi-task learning systems, developing novel architectures that improve cross-task generalization by 40%\\n- Pioneered new approach to zero-shot learning using contrastive training, published in NeurIPS 2023\\n- Built and led team of 6 researchers working on foundational ML models\\n- Developed novel regularization techniques for large language models, reducing catastrophic forgetting by 35%',\n",
       "   'start_date': '2019',\n",
       "   'end_date': 'Present'},\n",
       "  {'company': 'Google Research',\n",
       "   'title': 'Research Scientist',\n",
       "   'description': '- Developed probabilistic frameworks for robust ML, published in ICML 2018\\n- Created novel attention mechanisms for computer vision models, improving accuracy by 25%\\n- Led collaboration with Google Brain team on efficient training methods for transformer models\\n- Mentored 4 PhD interns and collaborated with academic institutions',\n",
       "   'start_date': '2015',\n",
       "   'end_date': '2019'},\n",
       "  {'company': 'Columbia University',\n",
       "   'title': 'Research Assistant Professor',\n",
       "   'description': '- Published seminal work on Bayesian optimization methods (cited 1000+ times)\\n- Taught graduate-level courses in Machine Learning and Statistical Learning Theory\\n- Supervised 5 PhD students and 3 MSc students\\n- Secured $500K in research grants for probabilistic ML research',\n",
       "   'start_date': '2011',\n",
       "   'end_date': '2015'}],\n",
       " 'education': [{'institution': 'Columbia University',\n",
       "   'degree': 'Ph.D. in Computer Science',\n",
       "   'start_date': '2007',\n",
       "   'end_date': '2011'},\n",
       "  {'institution': 'Stanford University',\n",
       "   'degree': 'M.S. in Computer Science',\n",
       "   'start_date': '2005',\n",
       "   'end_date': '2007'}]}"
      ]
     },
     "execution_count": null,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "agent.data_schema = Resume\n",
    "resume = agent.extract(\"./data/resumes/ai_researcher.pdf\")\n",
    "resume.data"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "This is a good start. Let us add a few more fields to the schema and re-run the extraction. "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "class TechnicalSkills(BaseModel):\n",
    "    programming_languages: List[str] = Field(\n",
    "        description=\"The programming languages the candidate is proficient in.\"\n",
    "    )\n",
    "    frameworks: List[str] = Field(\n",
    "        description=\"The tools/frameworks the candidate is proficient in, e.g. React, Django, PyTorch, etc.\"\n",
    "    )\n",
    "    skills: List[str] = Field(\n",
    "        description=\"Other general skills the candidate is proficient in, e.g. Data Engineering, Machine Learning, etc.\"\n",
    "    )\n",
    "\n",
    "\n",
    "class Resume(BaseModel):\n",
    "    name: str = Field(description=\"The name of the candidate\")\n",
    "    email: str = Field(description=\"The email address of the candidate\")\n",
    "    links: List[str] = Field(\n",
    "        description=\"The links to the candidate's social media profiles\"\n",
    "    )\n",
    "    experience: List[Experience] = Field(description=\"The candidate's experience\")\n",
    "    education: List[Education] = Field(description=\"The candidate's education\")\n",
    "    technical_skills: TechnicalSkills = Field(\n",
    "        description=\"The candidate's technical skills\"\n",
    "    )\n",
    "    key_accomplishments: str = Field(\n",
    "        description=\"Summarize the candidates highest achievements.\"\n",
    "    )"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "{'name': 'Dr. Rachel Zhang, Ph.D.',\n",
       " 'email': 'rachel.zhang@email.com',\n",
       " 'links': ['linkedin.com/in/rachelzhang',\n",
       "  'github.com/rzhang-ai',\n",
       "  'scholar.google.com/rachelzhang'],\n",
       " 'experience': [{'company': 'DeepMind',\n",
       "   'title': 'Senior Research Scientist',\n",
       "   'description': 'Lead researcher on large-scale multi-task learning systems, developing novel architectures that improve cross-task generalization by 40%\\nPioneered new approach to zero-shot learning using contrastive training, published in NeurIPS 2023\\nBuilt and led team of 6 researchers working on foundational ML models\\nDeveloped novel regularization techniques for large language models, reducing catastrophic forgetting by 35%',\n",
       "   'start_date': '2019',\n",
       "   'end_date': 'Present'},\n",
       "  {'company': 'Google Research',\n",
       "   'title': 'Research Scientist',\n",
       "   'description': 'Developed probabilistic frameworks for robust ML, published in ICML 2018\\nCreated novel attention mechanisms for computer vision models, improving accuracy by 25%\\nLed collaboration with Google Brain team on efficient training methods for transformer models\\nMentored 4 PhD interns and collaborated with academic institutions',\n",
       "   'start_date': '2015',\n",
       "   'end_date': '2019'},\n",
       "  {'company': 'Columbia University',\n",
       "   'title': 'Research Assistant Professor',\n",
       "   'description': 'Published seminal work on Bayesian optimization methods (cited 1000+ times)\\nTaught graduate-level courses in Machine Learning and Statistical Learning Theory\\nSupervised 5 PhD students and 3 MSc students\\nSecured $500K in research grants for probabilistic ML research',\n",
       "   'start_date': '2011',\n",
       "   'end_date': '2015'}],\n",
       " 'education': [{'institution': 'Columbia University',\n",
       "   'degree': 'Ph.D. in Computer Science',\n",
       "   'start_date': '2007',\n",
       "   'end_date': '2011'},\n",
       "  {'institution': 'Stanford University',\n",
       "   'degree': 'M.S. in Computer Science',\n",
       "   'start_date': '2005',\n",
       "   'end_date': '2007'}],\n",
       " 'technical_skills': {'programming_languages': ['Python',\n",
       "   'C++',\n",
       "   'Julia',\n",
       "   'CUDA'],\n",
       "  'frameworks': ['PyTorch', 'TensorFlow', 'JAX', 'Ray'],\n",
       "  'skills': ['Deep Learning',\n",
       "   'Reinforcement Learning',\n",
       "   'Probabilistic Models',\n",
       "   'Multi-Task Learning',\n",
       "   'Zero-Shot Learning',\n",
       "   'Neural Architecture Search']},\n",
       " 'key_accomplishments': 'AI researcher with 12+ years of experience spanning classical machine learning, deep learning, and probabilistic modeling. Led groundbreaking research in reinforcement learning, generative models, and multi-task learning. Published 25+ papers in top-tier conferences (NeurIPS, ICML, ICLR). Strong track record of transitioning theoretical advances into practical applications in both academic and industrial settings.'}"
      ]
     },
     "execution_count": null,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "agent.data_schema = Resume\n",
    "resume = agent.extract(\"./data/resumes/ai_researcher.pdf\")\n",
    "resume.data"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Finalizing the schema\n",
    "\n",
    "This is great! We have extracted a lot of key information from the resume that is well-typed and can be used downstream for further processing. Until now, this data is ephemeral and will be lost if we close the session. Let us save the state of our extraction and use it to extract data from multiple resumes. "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "agent.save()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "{'type': 'object',\n",
       " 'required': ['name',\n",
       "  'email',\n",
       "  'links',\n",
       "  'experience',\n",
       "  'education',\n",
       "  'technical_skills',\n",
       "  'key_accomplishments'],\n",
       " 'properties': {'name': {'type': 'string',\n",
       "   'description': 'The name of the candidate'},\n",
       "  'email': {'type': 'string',\n",
       "   'description': 'The email address of the candidate'},\n",
       "  'links': {'type': 'array',\n",
       "   'items': {'type': 'string'},\n",
       "   'description': \"The links to the candidate's social media profiles\"},\n",
       "  'education': {'type': 'array',\n",
       "   'items': {'type': 'object',\n",
       "    'required': ['institution', 'degree', 'start_date', 'end_date'],\n",
       "    'properties': {'degree': {'type': 'string',\n",
       "      'description': 'The degree of the candidate'},\n",
       "     'end_date': {'anyOf': [{'type': 'string'}, {'type': 'null'}],\n",
       "      'description': \"The end date of the candidate's education\"},\n",
       "     'start_date': {'anyOf': [{'type': 'string'}, {'type': 'null'}],\n",
       "      'description': \"The start date of the candidate's education\"},\n",
       "     'institution': {'type': 'string',\n",
       "      'description': 'The institution of the candidate'}},\n",
       "    'additionalProperties': False},\n",
       "   'description': \"The candidate's education\"},\n",
       "  'experience': {'type': 'array',\n",
       "   'items': {'type': 'object',\n",
       "    'required': ['company', 'title', 'description', 'start_date', 'end_date'],\n",
       "    'properties': {'title': {'type': 'string',\n",
       "      'description': 'The title of the candidate'},\n",
       "     'company': {'type': 'string', 'description': 'The name of the company'},\n",
       "     'end_date': {'anyOf': [{'type': 'string'}, {'type': 'null'}],\n",
       "      'description': \"The end date of the candidate's experience\"},\n",
       "     'start_date': {'anyOf': [{'type': 'string'}, {'type': 'null'}],\n",
       "      'description': \"The start date of the candidate's experience\"},\n",
       "     'description': {'anyOf': [{'type': 'string'}, {'type': 'null'}],\n",
       "      'description': \"The description of the candidate's experience\"}},\n",
       "    'additionalProperties': False},\n",
       "   'description': \"The candidate's experience\"},\n",
       "  'technical_skills': {'type': 'object',\n",
       "   'required': ['programming_languages', 'frameworks', 'skills'],\n",
       "   'properties': {'skills': {'type': 'array',\n",
       "     'items': {'type': 'string'},\n",
       "     'description': 'Other general skills the candidate is proficient in, e.g. Data Engineering, Machine Learning, etc.'},\n",
       "    'frameworks': {'type': 'array',\n",
       "     'items': {'type': 'string'},\n",
       "     'description': 'The tools/frameworks the candidate is proficient in, e.g. React, Django, PyTorch, etc.'},\n",
       "    'programming_languages': {'type': 'array',\n",
       "     'items': {'type': 'string'},\n",
       "     'description': 'The programming languages the candidate is proficient in.'}},\n",
       "   'description': \"The candidate's technical skills\",\n",
       "   'additionalProperties': False},\n",
       "  'key_accomplishments': {'type': 'string',\n",
       "   'description': 'Summarize the candidates highest achievements.'}},\n",
       " 'additionalProperties': False}"
      ]
     },
     "execution_count": null,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "agent = llama_extract.get_agent(\"resume-screening\")\n",
    "agent.data_schema  # Latest schema should be returned"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Queueing extractions"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "For multiple resumes, we can use the `queue_extraction` method to run extractions asynchronously. This is ideal for processing batch extraction jobs."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Uploading files: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3/3 [00:01<00:00,  2.13it/s]\n",
      "Creating extraction jobs: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3/3 [00:00<00:00,  5.83it/s]\n"
     ]
    }
   ],
   "source": [
    "import os\n",
    "\n",
    "# All resumes in the data/resumes directory\n",
    "resumes = []\n",
    "\n",
    "with os.scandir(\"./data/resumes\") as entries:\n",
    "    for entry in entries:\n",
    "        if entry.is_file():\n",
    "            resumes.append(entry.path)\n",
    "\n",
    "jobs = await agent.queue_extraction(resumes)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "To get the latest status of the extractions for any `job_id`, we can use the `get_extraction_job` method. \n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[<StatusEnum.PENDING: 'PENDING'>,\n",
       " <StatusEnum.PENDING: 'PENDING'>,\n",
       " <StatusEnum.PENDING: 'PENDING'>]"
      ]
     },
     "execution_count": null,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "[agent.get_extraction_job(job_id=job.id).status for job in jobs]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "We notice that all extraction runs are in a PENDING state. We can check back again to see if the extractions have completed. "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[<StatusEnum.SUCCESS: 'SUCCESS'>,\n",
       " <StatusEnum.SUCCESS: 'SUCCESS'>,\n",
       " <StatusEnum.SUCCESS: 'SUCCESS'>]"
      ]
     },
     "execution_count": null,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "[agent.get_extraction_job(job_id=job.id).status for job in jobs]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Retrieving results\n",
    "\n",
    "Let us now retrieve the results of the extractions. If the status of the extraction is `SUCCESS`, we can retrieve the data from the `data` field. In case there are errors (status = `ERROR`), we can retrieve the error message from the `error` field. \n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "results = []\n",
    "for job in jobs:\n",
    "    extract_run = agent.get_extraction_run_for_job(job.id)\n",
    "    if extract_run.status == \"SUCCESS\":\n",
    "        results.append(extract_run.data)\n",
    "    else:\n",
    "        print(f\"Extraction status for job {job.id}: {extract_run.status}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "{'name': 'Dr. Rachel Zhang, Ph.D.',\n",
       " 'email': 'rachel.zhang@email.com',\n",
       " 'links': ['linkedin.com/in/rachelzhang',\n",
       "  'github.com/rzhang-ai',\n",
       "  'scholar.google.com/rachelzhang'],\n",
       " 'education': [{'degree': 'Ph.D. in Computer Science',\n",
       "   'end_date': '2011',\n",
       "   'start_date': '2007',\n",
       "   'institution': 'Columbia University'},\n",
       "  {'degree': 'M.S. in Computer Science',\n",
       "   'end_date': '2007',\n",
       "   'start_date': '2005',\n",
       "   'institution': 'Stanford University'}],\n",
       " 'experience': [{'title': 'Senior Research Scientist',\n",
       "   'company': 'DeepMind',\n",
       "   'end_date': None,\n",
       "   'start_date': '2019',\n",
       "   'description': '- Lead researcher on large-scale multi-task learning systems, developing novel architectures that improve cross-task generalization by 40%\\n- Pioneered new approach to zero-shot learning using contrastive training, published in NeurIPS 2023\\n- Built and led team of 6 researchers working on foundational ML models\\n- Developed novel regularization techniques for large language models, reducing catastrophic forgetting by 35%'},\n",
       "  {'title': 'Research Scientist',\n",
       "   'company': 'Google Research',\n",
       "   'end_date': '2019',\n",
       "   'start_date': '2015',\n",
       "   'description': '- Developed probabilistic frameworks for robust ML, published in ICML 2018\\n- Created novel attention mechanisms for computer vision models, improving accuracy by 25%\\n- Led collaboration with Google Brain team on efficient training methods for transformer models\\n- Mentored 4 PhD interns and collaborated with academic institutions'},\n",
       "  {'title': 'Research Assistant Professor',\n",
       "   'company': 'Columbia University',\n",
       "   'end_date': '2015',\n",
       "   'start_date': '2011',\n",
       "   'description': '- Published seminal work on Bayesian optimization methods (cited 1000+ times)\\n- Taught graduate-level courses in Machine Learning and Statistical Learning Theory\\n- Supervised 5 PhD students and 3 MSc students\\n- Secured $500K in research grants for probabilistic ML research'}],\n",
       " 'technical_skills': {'skills': ['Deep Learning',\n",
       "   'Reinforcement Learning',\n",
       "   'Probabilistic Models',\n",
       "   'Multi-Task Learning',\n",
       "   'Zero-Shot Learning',\n",
       "   'Neural Architecture Search'],\n",
       "  'frameworks': ['PyTorch', 'TensorFlow', 'JAX', 'Ray'],\n",
       "  'programming_languages': ['Python', 'C++', 'Julia', 'CUDA']},\n",
       " 'key_accomplishments': 'AI researcher with 12+ years of experience spanning classical machine learning, deep learning, and probabilistic modeling. Led groundbreaking research in reinforcement learning, generative models, and multi-task learning. Published 25+ papers in top-tier conferences (NeurIPS, ICML, ICLR). Strong track record of transitioning theoretical advances into practical applications in both academic and industrial settings.'}"
      ]
     },
     "execution_count": null,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "results[0]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "{'name': 'Alex Park',\n",
       " 'email': 'alex park@email.com',\n",
       " 'links': ['linkedin.com/in/alexpark'],\n",
       " 'education': [{'degree': 'M.S. Computer Science',\n",
       "   'end_date': None,\n",
       "   'start_date': None,\n",
       "   'institution': 'University of California, Berkeley'},\n",
       "  {'degree': 'B.S. Computer Science',\n",
       "   'end_date': None,\n",
       "   'start_date': None,\n",
       "   'institution': 'University of California, Berkeley'}],\n",
       " 'experience': [{'title': 'Senior Machine Learning Engineer',\n",
       "   'company': 'SearchTech AI',\n",
       "   'end_date': None,\n",
       "   'start_date': None,\n",
       "   'description': 'Led development of next-generation learning-to-rank system using BER\\nArchitected and deployed real-time personalization system processing 10\\nIncreasing CTR by 15%\\nImproving search relevance by 24% (NDCG@10)'},\n",
       "  {'title': '',\n",
       "   'company': 'Commerce Corp',\n",
       "   'end_date': None,\n",
       "   'start_date': None,\n",
       "   'description': 'Developed semantic search system using transformer models and approximate nearest neighbors, reducing null search results by 35%'},\n",
       "  {'title': 'Machine Learning Engineer',\n",
       "   'company': 'Tech Solutions Inc',\n",
       "   'end_date': None,\n",
       "   'start_date': None,\n",
       "   'description': 'Implemented query understanding pipeline'},\n",
       "  {'title': 'Software Engineer',\n",
       "   'company': '',\n",
       "   'end_date': None,\n",
       "   'start_date': None,\n",
       "   'description': 'Built data pipelines and Flasticsearch'}],\n",
       " 'technical_skills': {'skills': ['Elasticsearch',\n",
       "   'Solr',\n",
       "   'Lucene',\n",
       "   'Python',\n",
       "   'SQL',\n",
       "   'Java',\n",
       "   'Scala',\n",
       "   'Shell Scripting'],\n",
       "  'frameworks': ['PyTorch',\n",
       "   'TensorFlow',\n",
       "   'Scikit-learn',\n",
       "   'BERT',\n",
       "   'Word2Vec',\n",
       "   'FastAI',\n",
       "   'BM25',\n",
       "   'FAISS',\n",
       "   'Docker',\n",
       "   'Kubernetes'],\n",
       "  'programming_languages': []},\n",
       " 'key_accomplishments': 'Machine Learning Engineer with 5 years of experience building and deploying large-scale search and relevance systems: Specialized in developing personalized search algorithms, learning-to-rank models; and recommendation systems. Strong track record of improving search relevance metrics and user engagement through ML-driven solutions:'}"
      ]
     },
     "execution_count": null,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "results[1]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "{'name': 'Sarah Chen',\n",
       " 'email': 'sarah.chen@email.com',\n",
       " 'links': [],\n",
       " 'education': [{'degree': 'Master of Science in Computer Science',\n",
       "   'end_date': '2013',\n",
       "   'start_date': None,\n",
       "   'institution': 'Stanford University'},\n",
       "  {'degree': 'Bachelor of Science in Computer Engineering',\n",
       "   'end_date': '2011',\n",
       "   'start_date': None,\n",
       "   'institution': 'University of California, Berkeley'}],\n",
       " 'experience': [{'title': 'Senior Software Architect',\n",
       "   'company': 'TechCorp Solutions',\n",
       "   'end_date': None,\n",
       "   'start_date': '2020',\n",
       "   'description': '- Led architectural design and implementation of a cloud-native platform serving 2M+ users\\n- Established architectural guidelines and best practices adopted across 12 development teams\\n- Reduced system latency by 40% through implementation of event-driven architecture\\n- Mentored 15+ senior developers in cloud-native development practices'},\n",
       "  {'title': 'Lead Software Engineer',\n",
       "   'company': 'DataFlow Systems',\n",
       "   'end_date': '2020',\n",
       "   'start_date': '2016',\n",
       "   'description': '- Architected and led development of distributed data processing platform handling 5TB daily\\n- Designed microservices architecture reducing deployment time by 65%\\n- Led migration of legacy monolith to cloud-native architecture\\n- Managed team of 8 engineers across 3 international locations'},\n",
       "  {'title': 'Senior Software Engineer',\n",
       "   'company': 'InnovateTech',\n",
       "   'end_date': '2016',\n",
       "   'start_date': '2013',\n",
       "   'description': '- Developed high-performance trading platform processing 100K transactions per second\\n- Implemented real-time analytics engine reducing processing latency by 75%\\n- Led adoption of container orchestration reducing deployment costs by 35%'}],\n",
       " 'technical_skills': {'skills': ['Architecture & Design',\n",
       "   'Microservices',\n",
       "   'Event-Driven Architecture',\n",
       "   'Domain-Driven Design',\n",
       "   'REST APIs',\n",
       "   'Cloud Platforms'],\n",
       "  'frameworks': ['AWS (Advanced)', 'Azure', 'Google Cloud Platform'],\n",
       "  'programming_languages': ['Java', 'Python', 'Go', 'JavaScript/TypeScript']},\n",
       " 'key_accomplishments': '- Co-inventor on three patents for distributed systems architecture\\n- Published paper on \"Scalable Microservices Architecture\" at IEEE Cloud Computing Conference 2022\\n- Keynote Speaker, CloudCon 2023: \"Future of Cloud-Native Architecture\"\\n- Regular presenter at local tech meetups and conferences'}"
      ]
     },
     "execution_count": null,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "results[2]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Congratulations! You now have an agent that can extract structured data from resumes. \n",
    "- You can now use this agent to extract data from more resumes and use the extracted data for further processing. \n",
    "- To update the schema, you can simply update the `data_schema` attribute of the agent and re-run the extraction. \n",
    "- You can also use the `save` method to save the state of the agent and persist changes to the schema for future use. \n",
    "\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}

{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "00f6713b-2a32-4f8f-80e5-9a7d9b6e3b90",
   "metadata": {},
   "source": [
    "# Solar Panel Datasheet Comparison Workflow\n",
    "\n",
    "<a href=\"https://colab.research.google.com/github/run-llama/llama_cloud_services/blob/main/examples/extract/solar_panel_e2e_comparison.ipynb\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>\n",
    "\n",
    "\n",
    "This notebook demonstrates an end‑to‑end agentic workflow using LlamaExtract and the LlamaIndex event‑driven workflow framework. In this workflow, we:\n",
    "\n",
    "1. **Extract** structured technical specifications from a solar panel datasheet (e.g. a PDF downloaded from a vendor).\n",
    "2. **Load** design requirements (provided as a text blob) for a lab‑grade solar panel.\n",
    "3. **Generate** a detailed comparison report by triggering an event that injects both the extracted data and the requirements into an LLM prompt.\n",
    "\n",
    "The workflow is designed for renewable energy engineers who need to quickly validate that a solar panel meets specific design criteria.\n",
    "\n",
    "The following notebook uses the event‑driven syntax (with custom events, steps, and a workflow class) adapted from the technical datasheet and contract review examples."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "ab7be988",
   "metadata": {},
   "source": [
    "> **⚠️ DEPRECATION NOTICE**>> This example uses the deprecated `llama-cloud-services` package, which will be maintained until **May 1, 2026**.>> **Please migrate to:**> - **Python**: `pip install llama-cloud>=1.0` ([GitHub](https://github.com/run-llama/llama-cloud-py))> - **New Package Documentation**: https://docs.cloud.llamaindex.ai/>> The new package provides the same functionality with improved performance and support."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "36d8e34e-ed98-46ac-b744-1642f6e253d5",
   "metadata": {},
   "source": [
    "## Setup and Load Data\n",
    "\n",
    "We download the [Honey M TSM-DE08M.08(II) datasheet](https://static.trinasolar.com/sites/default/files/EU_Datasheet_HoneyM_DE08M.08%28II%29_2021_A.pdf) as a PDF.\n",
    "\n",
    "**NOTE**: The design requirements are already stored in `data/solar_panel_e2e_comparison/design_reqs.txt`."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "1de7b1b3-c285-492c-8b2e-b37974b4fc63",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "--2025-04-01 14:47:56--  https://static.trinasolar.com/sites/default/files/EU_Datasheet_HoneyM_DE08M.08%28II%29_2021_A.pdf\n",
      "Resolving static.trinasolar.com (static.trinasolar.com)... 47.246.23.232, 47.246.23.234, 47.246.23.227, ...\n",
      "Connecting to static.trinasolar.com (static.trinasolar.com)|47.246.23.232|:443... connected.\n",
      "WARNING: cannot verify static.trinasolar.com's certificate, issued by ‘CN=DigiCert Global G2 TLS RSA SHA256 2020 CA1,O=DigiCert Inc,C=US’:\n",
      "  Unable to locally verify the issuer's authority.\n",
      "HTTP request sent, awaiting response... 200 OK\n",
      "Length: 1888183 (1.8M) [application/pdf]\n",
      "Saving to: ‘data/solar_panel_e2e_comparison/datasheet.pdf’\n",
      "\n",
      "data/solar_panel_e2 100%[===================>]   1.80M  7.47MB/s    in 0.2s    \n",
      "\n",
      "2025-04-01 14:47:56 (7.47 MB/s) - ‘data/solar_panel_e2e_comparison/datasheet.pdf’ saved [1888183/1888183]\n",
      "\n"
     ]
    }
   ],
   "source": [
    "!wget https://static.trinasolar.com/sites/default/files/EU_Datasheet_HoneyM_DE08M.08%28II%29_2021_A.pdf -O data/solar_panel_e2e_comparison/datasheet.pdf --no-check-certificate"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "89d2f4c9-f785-424d-a409-3381796c457c",
   "metadata": {},
   "source": [
    "## Define the Structured Extraction Schema\n",
    "\n",
    "We define a new, rich schema called `SolarPanelSchema` to capture key technical details from the datasheet. This schema includes:\n",
    "\n",
    "- **PowerRange:** Structured as minimum and maximum power output (in Watts).\n",
    "- **SolarPanelSpec:** Includes module name, power output range, maximum efficiency, certifications, and a mapping of page citations.\n",
    "\n",
    "This schema replaces the earlier LM317 schema and will be used when creating our extraction agent."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "bfb40d48-36e0-4b1c-97a1-32a1704c582b",
   "metadata": {},
   "outputs": [],
   "source": [
    "from pydantic import BaseModel, Field\n",
    "from typing import List\n",
    "\n",
    "\n",
    "class PowerRange(BaseModel):\n",
    "    min_power: float = Field(..., description=\"Minimum power output in Watts\")\n",
    "    max_power: float = Field(..., description=\"Maximum power output in Watts\")\n",
    "    unit: str = Field(\"W\", description=\"Power unit\")\n",
    "\n",
    "\n",
    "class SolarPanelSpec(BaseModel):\n",
    "    module_name: str = Field(..., description=\"Name or model of the solar panel module\")\n",
    "    power_output: PowerRange = Field(..., description=\"Power output range\")\n",
    "    maximum_efficiency: float = Field(\n",
    "        ..., description=\"Maximum module efficiency in percentage\"\n",
    "    )\n",
    "    temperature_coefficient: float = Field(\n",
    "        ..., description=\"Temperature coefficient in %/°C\"\n",
    "    )\n",
    "    certifications: List[str] = Field([], description=\"List of certifications\")\n",
    "    page_citations: dict = Field(\n",
    "        ..., description=\"Mapping of each extracted field to its page numbers\"\n",
    "    )\n",
    "\n",
    "\n",
    "class SolarPanelSchema(BaseModel):\n",
    "    specs: List[SolarPanelSpec] = Field(\n",
    "        ..., description=\"List of extracted solar panel specifications\"\n",
    "    )"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "19dc309e-7cec-43c1-8f6c-72e14df58f8f",
   "metadata": {},
   "source": [
    "## Initialize Extraction Agent\n",
    "\n",
    "Here we initialize our extraction agent that will be responsible for extracting the schema from the solar panel datasheet."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "c9d9f4a2-2e14-493d-8a7e-d01159d38b8f",
   "metadata": {},
   "outputs": [],
   "source": [
    "from dotenv import load_dotenv\n",
    "from llama_cloud_services import LlamaExtract\n",
    "from llama_cloud.core.api_error import ApiError\n",
    "from llama_cloud import ExtractConfig\n",
    "\n",
    "# Initialize the LlamaExtract client\n",
    "llama_extract = LlamaExtract(\n",
    "    project_id=\"2fef999e-1073-40e6-aeb3-1f3c0e64d99b\",\n",
    "    organization_id=\"43b88c8f-e488-46f6-9013-698e3d2e374a\",\n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "ec0eb2a7-6e02-45da-a6af-227e2f7c81f2",
   "metadata": {},
   "outputs": [],
   "source": [
    "try:\n",
    "    existing_agent = llama_extract.get_agent(name=\"solar-panel-datasheet\")\n",
    "    if existing_agent:\n",
    "        llama_extract.delete_agent(existing_agent.id)\n",
    "except ApiError as e:\n",
    "    if e.status_code == 404:\n",
    "        pass\n",
    "    else:\n",
    "        raise\n",
    "\n",
    "extract_config = ExtractConfig(\n",
    "    extraction_mode=\"BALANCED\",\n",
    ")\n",
    "\n",
    "agent = llama_extract.create_agent(\n",
    "    name=\"solar-panel-datasheet\", data_schema=SolarPanelSchema, config=extract_config\n",
    ")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "b4d7bb60-0456-4a2d-8d48-14f9bb3e71d2",
   "metadata": {},
   "source": [
    "## Workflow Overview\n",
    "\n",
    "The workflow consists of four main steps:\n",
    "\n",
    "1. **parse_datasheet:** Reads the solar panel datasheet (PDF) and converts its content into text (with page citations).\n",
    "2. **load_requirements:** Loads the design requirements (as a text blob) that will be injected into the prompt.\n",
    "3. **generate_comparison_report:** Constructs a prompt using the extracted datasheet content and design requirements and triggers the LLM to generate a comparison report.\n",
    "4. **output_result:** Logs and returns the final report as the workflow’s result.\n",
    "\n",
    "Each step is implemented as an asynchronous function decorated with `@step`, and the workflow is built by subclassing `Workflow`."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "7c482e3a-66b4-4e1b-8d2d-9a9c6b3967f3",
   "metadata": {},
   "outputs": [],
   "source": [
    "from llama_index.core.workflow import (\n",
    "    Event,\n",
    "    StartEvent,\n",
    "    StopEvent,\n",
    "    Context,\n",
    "    Workflow,\n",
    "    step,\n",
    ")\n",
    "from llama_index.llms.openai import OpenAI\n",
    "from llama_index.core.prompts import ChatPromptTemplate\n",
    "from llama_cloud_services import LlamaExtract\n",
    "from llama_cloud.core.api_error import ApiError\n",
    "from pydantic import BaseModel, Field\n",
    "from typing import List\n",
    "\n",
    "\n",
    "# Define output schema for the comparison report (for reference)\n",
    "class ComparisonReportOutput(BaseModel):\n",
    "    component_name: str = Field(\n",
    "        ..., description=\"The name of the component being evaluated.\"\n",
    "    )\n",
    "    meets_requirements: bool = Field(\n",
    "        ...,\n",
    "        description=\"Overall indicator of whether the component meets the design criteria.\",\n",
    "    )\n",
    "    summary: str = Field(..., description=\"A brief summary of the evaluation results.\")\n",
    "    details: dict = Field(\n",
    "        ..., description=\"Detailed comparisons for each key parameter.\"\n",
    "    )\n",
    "\n",
    "\n",
    "# Define custom events\n",
    "\n",
    "\n",
    "class DatasheetParseEvent(Event):\n",
    "    datasheet_content: dict\n",
    "\n",
    "\n",
    "class RequirementsLoadEvent(Event):\n",
    "    requirements_text: str\n",
    "\n",
    "\n",
    "class ComparisonReportEvent(Event):\n",
    "    report: ComparisonReportOutput\n",
    "\n",
    "\n",
    "class LogEvent(Event):\n",
    "    msg: str\n",
    "    delta: bool = False\n",
    "\n",
    "\n",
    "# For our demonstration, we assume that LlamaExtract is used to parse the datasheet into text.\n",
    "# We'll also use OpenAI (via LlamaIndex) as our LLM for generating the report.\n",
    "\n",
    "llm = OpenAI(model=\"gpt-4o\")  # or your preferred model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "67a0c391-c7f5-4b93-8d6b-9e31b2d7a817",
   "metadata": {},
   "outputs": [],
   "source": [
    "class SolarPanelComparisonWorkflow(Workflow):\n",
    "    \"\"\"\n",
    "    Workflow to extract data from a solar panel datasheet and generate a comparison report\n",
    "    against provided design requirements.\n",
    "    \"\"\"\n",
    "\n",
    "    def __init__(self, agent: LlamaExtract, requirements_path: str, **kwargs):\n",
    "        super().__init__(**kwargs)\n",
    "        self.agent = agent\n",
    "        # Load design requirements from file as a text blob\n",
    "        with open(requirements_path, \"r\") as f:\n",
    "            self.requirements_text = f.read()\n",
    "\n",
    "    @step\n",
    "    async def parse_datasheet(\n",
    "        self, ctx: Context, ev: StartEvent\n",
    "    ) -> DatasheetParseEvent:\n",
    "        # datasheet_path is provided in the StartEvent\n",
    "        datasheet_path = (\n",
    "            ev.datasheet_path\n",
    "        )  # e.g., \"./data/solar_panel_comparison/datasheet.pdf\"\n",
    "        extraction_result = await self.agent.aextract(datasheet_path)\n",
    "        datasheet_dict = (\n",
    "            extraction_result.data\n",
    "        )  # assumed to be a string with page citations\n",
    "        await ctx.set(\"datasheet_content\", datasheet_dict)\n",
    "        ctx.write_event_to_stream(LogEvent(msg=\"Datasheet parsed successfully.\"))\n",
    "        return DatasheetParseEvent(datasheet_content=datasheet_dict)\n",
    "\n",
    "    @step\n",
    "    async def load_requirements(\n",
    "        self, ctx: Context, ev: DatasheetParseEvent\n",
    "    ) -> RequirementsLoadEvent:\n",
    "        # Use the pre-loaded requirements text from __init__\n",
    "        req_text = self.requirements_text\n",
    "        ctx.write_event_to_stream(LogEvent(msg=\"Design requirements loaded.\"))\n",
    "        return RequirementsLoadEvent(requirements_text=req_text)\n",
    "\n",
    "    @step\n",
    "    async def generate_comparison_report(\n",
    "        self, ctx: Context, ev: RequirementsLoadEvent\n",
    "    ) -> StopEvent:\n",
    "        # Build a prompt that injects both the extracted datasheet content and the design requirements\n",
    "        datasheet_content = await ctx.get(\"datasheet_content\")\n",
    "        prompt_str = \"\"\"\n",
    "You are an expert renewable energy engineer.\n",
    "\n",
    "Compare the following solar panel datasheet information with the design requirements.\n",
    "\n",
    "Design Requirements:\n",
    "{requirements_text}\n",
    "\n",
    "Extracted Datasheet Information:\n",
    "{datasheet_content}\n",
    "\n",
    "Generate a detailed comparison report in JSON format with the following schema:\n",
    "  - component_name: string\n",
    "  - meets_requirements: boolean\n",
    "  - summary: string\n",
    "  - details: dictionary of comparisons for each parameter\n",
    "\n",
    "For each parameter (Maximum Power, Open-Circuit Voltage, Short-Circuit Current, Efficiency, Temperature Coefficient),\n",
    "indicate PASS or FAIL and provide brief explanations and recommendations.\n",
    "\"\"\"\n",
    "\n",
    "        # extract from contract\n",
    "        prompt = ChatPromptTemplate.from_messages([(\"user\", prompt_str)])\n",
    "\n",
    "        # Call the LLM to generate the report using the prompt\n",
    "        report_output = await llm.astructured_predict(\n",
    "            ComparisonReportOutput,\n",
    "            prompt,\n",
    "            requirements_text=ev.requirements_text,\n",
    "            datasheet_content=str(datasheet_content),\n",
    "        )\n",
    "        ctx.write_event_to_stream(LogEvent(msg=\"Comparison report generated.\"))\n",
    "        return StopEvent(\n",
    "            result={\"report\": report_output, \"datasheet_content\": datasheet_content}\n",
    "        )"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "d205f532-1a11-4a48-b5a8-87a7f85e9ce7",
   "metadata": {},
   "source": [
    "## Running the Workflow\n",
    "\n",
    "Below, we instantiate and run the workflow. We inject the design requirements as a text blob (no custom code to load) and pass the path to the solar panel datasheet (the HoneyM datasheet from Trina).\n",
    "\n",
    "The design requirements are:\n",
    "\n",
    "```\n",
    "Solar Panel Design Requirements:\n",
    "- Power Output Range: ≥ 350 W\n",
    "- Maximum Efficiency: ≥ 18%\n",
    "- Certifications: Must include IEC61215 and UL1703\n",
    "```\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "6b24fa61-a2f5-4ebb-84eb-1c9b48683b1b",
   "metadata": {},
   "outputs": [],
   "source": [
    "import nest_asyncio\n",
    "\n",
    "nest_asyncio.apply()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "be3ebad5-1f70-4671-a2ec-17bf9e4d788f",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Path to design requirements file (e.g., a text file with design criteria for solar panels)\n",
    "requirements_path = \"./data/solar_panel_e2e_comparison/design_reqs.txt\"\n",
    "\n",
    "# Instantiate the workflow\n",
    "workflow = SolarPanelComparisonWorkflow(\n",
    "    agent=agent, requirements_path=requirements_path, verbose=True, timeout=120\n",
    ")\n",
    "\n",
    "# Run the workflow; pass the datasheet path in the StartEvent\n",
    "result = await workflow.run(\n",
    "    datasheet_path=\"./data/solar_panel_e2e_comparison/datasheet.pdf\"\n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "e1e61f1e-8701-4acc-8f99-cc89d8aae535",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "********Final Comparison Report:********\n",
      "\n",
      "{\n",
      "    \"component_name\": \"TSM-DE08M.08(II)\",\n",
      "    \"meets_requirements\": true,\n",
      "    \"summary\": \"The solar panel TSM-DE08M.08(II) meets all the design requirements, making it a suitable choice for the intended application.\",\n",
      "    \"details\": {\n",
      "        \"Maximum Power Output\": \"PASS - The panel's power output ranges from 360 W to 385 W, exceeding the minimum requirement of 350 W.\",\n",
      "        \"Open-Circuit Voltage\": \"PASS - The datasheet does not specify Voc, but the panel meets other critical requirements. Verification of Voc is recommended.\",\n",
      "        \"Short-Circuit Current\": \"PASS - The datasheet does not specify Isc, but the panel meets other critical requirements. Verification of Isc is recommended.\",\n",
      "        \"Efficiency\": \"PASS - The panel's efficiency is 21.0%, which is above the required 18%.\",\n",
      "        \"Temperature Coefficient\": \"PASS - The temperature coefficient is -0.34%/°C, which is better than the maximum allowable -0.5%/°C.\"\n",
      "    }\n",
      "}\n"
     ]
    }
   ],
   "source": [
    "print(\"\\n********Final Comparison Report:********\\n\")\n",
    "print(result[\"report\"].model_dump_json(indent=4))\n",
    "# print(\"\\n********Datasheet Content:********\\n\", result[\"datasheet_content\"])"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "llama_parse",
   "language": "python",
   "name": "llama_parse"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}






----- 
ollama

> ## Documentation Index
> Fetch the complete documentation index at: https://docs.ollama.com/llms.txt
> Use this file to discover all available pages before exploring further.

# Tool calling

Ollama supports tool calling (also known as function calling) which allows a model to invoke tools and incorporate their results into its replies.

## Calling a single tool

Invoke a single tool and include its response in a follow-up request.

Also known as "single-shot" tool calling.

<Tabs>
  <Tab title="cURL">
    ```shell  theme={"system"}
    curl -s http://localhost:11434/api/chat -H "Content-Type: application/json" -d '{
      "model": "qwen3",
      "messages": [{"role": "user", "content": "What is the temperature in New York?"}],
      "stream": false,
      "tools": [
        {
          "type": "function",
          "function": {
            "name": "get_temperature",
            "description": "Get the current temperature for a city",
            "parameters": {
              "type": "object",
              "required": ["city"],
              "properties": {
                "city": {"type": "string", "description": "The name of the city"}
              }
            }
          }
        }
      ]
    }'
    ```

    **Generate a response with a single tool result**

    ```shell  theme={"system"}
    curl -s http://localhost:11434/api/chat -H "Content-Type: application/json" -d '{
      "model": "qwen3",
      "messages": [
        {"role": "user", "content": "What is the temperature in New York?"},
        {
          "role": "assistant",
          "tool_calls": [
            {
              "type": "function",
              "function": {
                "index": 0,
                "name": "get_temperature",
                "arguments": {"city": "New York"}
              }
            }
          ]
        },
        {"role": "tool", "tool_name": "get_temperature", "content": "22°C"}
      ],
      "stream": false
    }'
    ```
  </Tab>

  <Tab title="Python">
    Install the Ollama Python SDK:

    ```bash  theme={"system"}
    # with pip
    pip install ollama -U

    # with uv
    uv add ollama    
    ```

    ```python  theme={"system"}
    from ollama import chat

    def get_temperature(city: str) -> str:
      """Get the current temperature for a city
      
      Args:
        city: The name of the city

      Returns:
        The current temperature for the city
      """
      temperatures = {
        "New York": "22°C",
        "London": "15°C",
        "Tokyo": "18°C",
      }
      return temperatures.get(city, "Unknown")

    messages = [{"role": "user", "content": "What is the temperature in New York?"}]

    # pass functions directly as tools in the tools list or as a JSON schema
    response = chat(model="qwen3", messages=messages, tools=[get_temperature], think=True)

    messages.append(response.message)
    if response.message.tool_calls:
      # only recommended for models which only return a single tool call
      call = response.message.tool_calls[0]
      result = get_temperature(**call.function.arguments)
      # add the tool result to the messages
      messages.append({"role": "tool", "tool_name": call.function.name, "content": str(result)})

      final_response = chat(model="qwen3", messages=messages, tools=[get_temperature], think=True)
      print(final_response.message.content)
    ```
  </Tab>

  <Tab title="JavaScript">
    Install the Ollama JavaScript library:

    ```bash  theme={"system"}
    # with npm
    npm i ollama

    # with bun
    bun i ollama
    ```

    ```typescript  theme={"system"}
    import ollama from 'ollama'

    function getTemperature(city: string): string {
      const temperatures: Record<string, string> = {
        'New York': '22°C',
        'London': '15°C',
        'Tokyo': '18°C',
      }
      return temperatures[city] ?? 'Unknown'
    }

    const tools = [
      {
        type: 'function',
        function: {
          name: 'get_temperature',
          description: 'Get the current temperature for a city',
          parameters: {
            type: 'object',
            required: ['city'],
            properties: {
              city: { type: 'string', description: 'The name of the city' },
            },
          },
        },
      },
    ]

    const messages = [{ role: 'user', content: "What is the temperature in New York?" }]

    const response = await ollama.chat({
      model: 'qwen3',
      messages,
      tools,
      think: true,
    })

    messages.push(response.message)
    if (response.message.tool_calls?.length) {
      // only recommended for models which only return a single tool call
      const call = response.message.tool_calls[0]
      const args = call.function.arguments as { city: string }
      const result = getTemperature(args.city)
      // add the tool result to the messages
      messages.push({ role: 'tool', tool_name: call.function.name, content: result })

      // generate the final response
      const finalResponse = await ollama.chat({ model: 'qwen3', messages, tools, think: true })
      console.log(finalResponse.message.content)
    }
    ```
  </Tab>
</Tabs>

## Parallel tool calling

<Tabs>
  <Tab title="cURL">
    Request multiple tool calls in parallel, then send all tool responses back to the model.

    ```shell  theme={"system"}
    curl -s http://localhost:11434/api/chat -H "Content-Type: application/json" -d '{
      "model": "qwen3",
      "messages": [{"role": "user", "content": "What are the current weather conditions and temperature in New York and London?"}],
      "stream": false,
      "tools": [
        {
          "type": "function",
          "function": {
            "name": "get_temperature",
            "description": "Get the current temperature for a city",
            "parameters": {
              "type": "object",
              "required": ["city"],
              "properties": {
                "city": {"type": "string", "description": "The name of the city"}
              }
            }
          }
        },
        {
          "type": "function",
          "function": {
            "name": "get_conditions",
            "description": "Get the current weather conditions for a city",
            "parameters": {
              "type": "object",
              "required": ["city"],
              "properties": {
                "city": {"type": "string", "description": "The name of the city"}
              }
            }
          }
        }
      ]
    }'
    ```

    **Generate a response with multiple tool results**

    ```shell  theme={"system"}
    curl -s http://localhost:11434/api/chat -H "Content-Type: application/json" -d '{
      "model": "qwen3",
      "messages": [
        {"role": "user", "content": "What are the current weather conditions and temperature in New York and London?"},
        {
          "role": "assistant",
          "tool_calls": [
            {
              "type": "function",
              "function": {
                "index": 0,
                "name": "get_temperature",
                "arguments": {"city": "New York"}
              }
            },
            {
              "type": "function",
              "function": {
                "index": 1,
                "name": "get_conditions",
                "arguments": {"city": "New York"}
              }
            },
            {
              "type": "function",
              "function": {
                "index": 2,
                "name": "get_temperature",
                "arguments": {"city": "London"}
              }
            },
            {
              "type": "function",
              "function": {
                "index": 3,
                "name": "get_conditions",
                "arguments": {"city": "London"}
              }
            }
          ]
        },
        {"role": "tool", "tool_name": "get_temperature", "content": "22°C"},
        {"role": "tool", "tool_name": "get_conditions", "content": "Partly cloudy"},
        {"role": "tool", "tool_name": "get_temperature", "content": "15°C"},
        {"role": "tool", "tool_name": "get_conditions", "content": "Rainy"}
      ],
      "stream": false
    }'
    ```
  </Tab>

  <Tab title="Python">
    ```python  theme={"system"}
    from ollama import chat

    def get_temperature(city: str) -> str:
      """Get the current temperature for a city
      
      Args:
        city: The name of the city

      Returns:
        The current temperature for the city
      """
      temperatures = {
        "New York": "22°C",
        "London": "15°C",
        "Tokyo": "18°C"
      }
      return temperatures.get(city, "Unknown")

    def get_conditions(city: str) -> str:
      """Get the current weather conditions for a city
      
      Args:
        city: The name of the city

      Returns:
        The current weather conditions for the city
      """
      conditions = {
        "New York": "Partly cloudy",
        "London": "Rainy",
        "Tokyo": "Sunny"
      }
      return conditions.get(city, "Unknown")


    messages = [{'role': 'user', 'content': 'What are the current weather conditions and temperature in New York and London?'}]

    # The python client automatically parses functions as a tool schema so we can pass them directly
    # Schemas can be passed directly in the tools list as well 
    response = chat(model='qwen3', messages=messages, tools=[get_temperature, get_conditions], think=True)

    # add the assistant message to the messages
    messages.append(response.message)
    if response.message.tool_calls:
      # process each tool call 
      for call in response.message.tool_calls:
        # execute the appropriate tool
        if call.function.name == 'get_temperature':
          result = get_temperature(**call.function.arguments)
        elif call.function.name == 'get_conditions':
          result = get_conditions(**call.function.arguments)
        else:
          result = 'Unknown tool'
        # add the tool result to the messages
        messages.append({'role': 'tool',  'tool_name': call.function.name, 'content': str(result)})

      # generate the final response
      final_response = chat(model='qwen3', messages=messages, tools=[get_temperature, get_conditions], think=True)
      print(final_response.message.content)
    ```
  </Tab>

  <Tab title="JavaScript">
    ```typescript  theme={"system"}
    import ollama from 'ollama'

    function getTemperature(city: string): string {
      const temperatures: { [key: string]: string } = {
        "New York": "22°C",
        "London": "15°C",
        "Tokyo": "18°C"
      }
      return temperatures[city] || "Unknown"
    }

    function getConditions(city: string): string {
      const conditions: { [key: string]: string } = {
        "New York": "Partly cloudy",
        "London": "Rainy",
        "Tokyo": "Sunny"
      }
      return conditions[city] || "Unknown"
    }

    const tools = [
      {
        type: 'function',
        function: {
          name: 'get_temperature',
          description: 'Get the current temperature for a city',
          parameters: {
            type: 'object',
            required: ['city'],
            properties: {
              city: { type: 'string', description: 'The name of the city' },
            },
          },
        },
      },
      {
        type: 'function',
        function: {
          name: 'get_conditions',
          description: 'Get the current weather conditions for a city',
          parameters: {
            type: 'object',
            required: ['city'],
            properties: {
              city: { type: 'string', description: 'The name of the city' },
            },
          },
        },
      }
    ]

    const messages = [{ role: 'user', content: 'What are the current weather conditions and temperature in New York and London?' }]

    const response = await ollama.chat({
      model: 'qwen3',
      messages,
      tools,
      think: true
    })

    // add the assistant message to the messages
    messages.push(response.message)
    if (response.message.tool_calls) {
      // process each tool call 
      for (const call of response.message.tool_calls) {
        // execute the appropriate tool
        let result: string
        if (call.function.name === 'get_temperature') {
          const args = call.function.arguments as { city: string }
          result = getTemperature(args.city)
        } else if (call.function.name === 'get_conditions') {
          const args = call.function.arguments as { city: string }
          result = getConditions(args.city)
        } else {
          result = 'Unknown tool'
        }
        // add the tool result to the messages
        messages.push({ role: 'tool', tool_name: call.function.name, content: result })
      }

      // generate the final response
      const finalResponse = await ollama.chat({ model: 'qwen3', messages, tools, think: true })
      console.log(finalResponse.message.content)
    }
    ```
  </Tab>
</Tabs>

## Multi-turn tool calling (Agent loop)

An agent loop allows the model to decide when to invoke tools and incorporate their results into its replies.

It also might help to tell the model that it is in a loop and can make multiple tool calls.

<Tabs>
  <Tab title="Python">
    ```python  theme={"system"}
    from ollama import chat, ChatResponse


    def add(a: int, b: int) -> int:
      """Add two numbers"""
      """
      Args:
        a: The first number
        b: The second number

      Returns:
        The sum of the two numbers
      """
      return a + b


    def multiply(a: int, b: int) -> int:
      """Multiply two numbers"""
      """
      Args:
        a: The first number
        b: The second number

      Returns:
        The product of the two numbers
      """
      return a * b


    available_functions = {
      'add': add,
      'multiply': multiply,
    }

    messages = [{'role': 'user', 'content': 'What is (11434+12341)*412?'}]
    while True:
        response: ChatResponse = chat(
            model='qwen3',
            messages=messages,
            tools=[add, multiply],
            think=True,
        )
        messages.append(response.message)
        print("Thinking: ", response.message.thinking)
        print("Content: ", response.message.content)
        if response.message.tool_calls:
            for tc in response.message.tool_calls:
                if tc.function.name in available_functions:
                    print(f"Calling {tc.function.name} with arguments {tc.function.arguments}")
                    result = available_functions[tc.function.name](**tc.function.arguments)
                    print(f"Result: {result}")
                    # add the tool result to the messages
                    messages.append({'role': 'tool', 'tool_name': tc.function.name, 'content': str(result)})
        else:
            # end the loop when there are no more tool calls
            break
      # continue the loop with the updated messages
    ```
  </Tab>

  <Tab title="JavaScript">
    ```typescript  theme={"system"}
    import ollama from 'ollama'

    type ToolName = 'add' | 'multiply'

    function add(a: number, b: number): number {
      return a + b
    }

    function multiply(a: number, b: number): number {
      return a * b
    }

    const availableFunctions: Record<ToolName, (a: number, b: number) => number> = {
      add,
      multiply,
    }

    const tools = [
      {
        type: 'function',
        function: {
          name: 'add',
          description: 'Add two numbers',
          parameters: {
            type: 'object',
            required: ['a', 'b'],
            properties: {
              a: { type: 'integer', description: 'The first number' },
              b: { type: 'integer', description: 'The second number' },
            },
          },
        },
      },
      {
        type: 'function',
        function: {
          name: 'multiply',
          description: 'Multiply two numbers',
          parameters: {
            type: 'object',
            required: ['a', 'b'],
            properties: {
              a: { type: 'integer', description: 'The first number' },
              b: { type: 'integer', description: 'The second number' },
            },
          },
        },
      },
    ]

    async function agentLoop() {
      const messages = [{ role: 'user', content: 'What is (11434+12341)*412?' }]

      while (true) {
        const response = await ollama.chat({
          model: 'qwen3',
          messages,
          tools,
          think: true,
        })

        messages.push(response.message)
        console.log('Thinking:', response.message.thinking)
        console.log('Content:', response.message.content)

        const toolCalls = response.message.tool_calls ?? []
        if (toolCalls.length) {
          for (const call of toolCalls) {
            const fn = availableFunctions[call.function.name as ToolName]
            if (!fn) {
              continue
            }

            const args = call.function.arguments as { a: number; b: number }
            console.log(`Calling ${call.function.name} with arguments`, args)
            const result = fn(args.a, args.b)
            console.log(`Result: ${result}`)
            messages.push({ role: 'tool', tool_name: call.function.name, content: String(result) })
          }
        } else {
          break
        }
      }
    }

    agentLoop().catch(console.error)
    ```
  </Tab>
</Tabs>

## Tool calling with streaming

When streaming, gather every chunk of `thinking`, `content`, and `tool_calls`, then return those fields together with any tool results in the follow-up request.

<Tabs>
  <Tab title="Python">
    ```python  theme={"system"}
    from ollama import chat 


    def get_temperature(city: str) -> str:
      """Get the current temperature for a city
      
      Args:
        city: The name of the city

      Returns:
        The current temperature for the city
      """
      temperatures = {
        'New York': '22°C',
        'London': '15°C',
      }
      return temperatures.get(city, 'Unknown')


    messages = [{'role': 'user', 'content': "What is the temperature in New York?"}]

    while True:
      stream = chat(
        model='qwen3',
        messages=messages,
        tools=[get_temperature],
        stream=True,
        think=True,
      )

      thinking = ''
      content = ''
      tool_calls = []

      done_thinking = False
      # accumulate the partial fields
      for chunk in stream:
        if chunk.message.thinking:
          thinking += chunk.message.thinking
          print(chunk.message.thinking, end='', flush=True)
        if chunk.message.content:
          if not done_thinking:
            done_thinking = True
            print('\n')
          content += chunk.message.content
          print(chunk.message.content, end='', flush=True)
        if chunk.message.tool_calls:
          tool_calls.extend(chunk.message.tool_calls)
          print(chunk.message.tool_calls)

      # append accumulated fields to the messages
      if thinking or content or tool_calls:
        messages.append({'role': 'assistant', 'thinking': thinking, 'content': content, 'tool_calls': tool_calls})

      if not tool_calls:
        break

      for call in tool_calls:
        if call.function.name == 'get_temperature':
          result = get_temperature(**call.function.arguments)
        else:
          result = 'Unknown tool'
        messages.append({'role': 'tool', 'tool_name': call.function.name, 'content': result})
    ```
  </Tab>

  <Tab title="JavaScript">
    ```typescript  theme={"system"}
    import ollama from 'ollama'

    function getTemperature(city: string): string {
      const temperatures: Record<string, string> = {
        'New York': '22°C',
        'London': '15°C',
      }
      return temperatures[city] ?? 'Unknown'
    }

    const getTemperatureTool = {
      type: 'function',
      function: {
        name: 'get_temperature',
        description: 'Get the current temperature for a city',
        parameters: {
          type: 'object',
          required: ['city'],
          properties: {
            city: { type: 'string', description: 'The name of the city' },
          },
        },
      },
    }

    async function agentLoop() {
      const messages = [{ role: 'user', content: "What is the temperature in New York?" }]

      while (true) {
        const stream = await ollama.chat({
          model: 'qwen3',
          messages,
          tools: [getTemperatureTool],
          stream: true,
          think: true,
        })

        let thinking = ''
        let content = ''
        const toolCalls: any[] = []
        let doneThinking = false

        for await (const chunk of stream) {
          if (chunk.message.thinking) {
            thinking += chunk.message.thinking
            process.stdout.write(chunk.message.thinking)
          }
          if (chunk.message.content) {
            if (!doneThinking) {
              doneThinking = true
              process.stdout.write('\n')
            }
            content += chunk.message.content
            process.stdout.write(chunk.message.content)
          }
          if (chunk.message.tool_calls?.length) {
            toolCalls.push(...chunk.message.tool_calls)
            console.log(chunk.message.tool_calls)
          }
        }

        if (thinking || content || toolCalls.length) {
          messages.push({ role: 'assistant', thinking, content, tool_calls: toolCalls } as any)
        }

        if (!toolCalls.length) {
          break
        }

        for (const call of toolCalls) {
          if (call.function.name === 'get_temperature') {
            const args = call.function.arguments as { city: string }
            const result = getTemperature(args.city)
            messages.push({ role: 'tool', tool_name: call.function.name, content: result } )
          } else {
            messages.push({ role: 'tool', tool_name: call.function.name, content: 'Unknown tool' } )
          }
        }
      }
    }

    agentLoop().catch(console.error)
    ```
  </Tab>
</Tabs>

This loop streams the assistant response, accumulates partial fields, passes them back together, and appends the tool results so the model can complete its answer.

## Using functions as tools with Ollama Python SDK

The Python SDK automatically parses functions as a tool schema so we can pass them directly.
Schemas can still be passed if needed.

```python  theme={"system"}
from ollama import chat

def get_temperature(city: str) -> str:
  """Get the current temperature for a city
  
  Args:
    city: The name of the city

  Returns:
    The current temperature for the city
  """
  temperatures = {
    'New York': '22°C',
    'London': '15°C',
  }
  return temperatures.get(city, 'Unknown')

available_functions = {
  'get_temperature': get_temperature,
}
# directly pass the function as part of the tools list
response = chat(model='qwen3', messages=messages, tools=available_functions.values(), think=True)
```

> ## Documentation Index
> Fetch the complete documentation index at: https://docs.ollama.com/llms.txt
> Use this file to discover all available pages before exploring further.

# Embeddings

> Generate text embeddings for semantic search, retrieval, and RAG.

Embeddings turn text into numeric vectors you can store in a vector database, search with cosine similarity, or use in RAG pipelines. The vector length depends on the model (typically 384–1024 dimensions).

## Recommended models

* [embeddinggemma](https://ollama.com/library/embeddinggemma)
* [qwen3-embedding](https://ollama.com/library/qwen3-embedding)
* [all-minilm](https://ollama.com/library/all-minilm)

## Generate embeddings

<Tabs>
  <Tab title="CLI">
    Generate embeddings directly from the command line:

    ```shell  theme={"system"}
    ollama run embeddinggemma "Hello world"
    ```

    You can also pipe text to generate embeddings:

    ```shell  theme={"system"}
    echo "Hello world" | ollama run embeddinggemma
    ```

    Output is a JSON array.
  </Tab>

  <Tab title="cURL">
    ```shell  theme={"system"}
    curl -X POST http://localhost:11434/api/embed \
      -H "Content-Type: application/json" \
      -d '{
        "model": "embeddinggemma",
        "input": "The quick brown fox jumps over the lazy dog."
      }'
    ```
  </Tab>

  <Tab title="Python">
    ```python  theme={"system"}
    import ollama

    single = ollama.embed(
      model='embeddinggemma',
      input='The quick brown fox jumps over the lazy dog.'
    )
    print(len(single['embeddings'][0]))  # vector length
    ```
  </Tab>

  <Tab title="JavaScript">
    ```javascript  theme={"system"}
    import ollama from 'ollama'

    const single = await ollama.embed({
      model: 'embeddinggemma',
      input: 'The quick brown fox jumps over the lazy dog.',
    })
    console.log(single.embeddings[0].length) // vector length
    ```
  </Tab>
</Tabs>

<Note>
  The `/api/embed` endpoint returns L2‑normalized (unit‑length) vectors.
</Note>

## Generate a batch of embeddings

Pass an array of strings to `input`.

<Tabs>
  <Tab title="cURL">
    ```shell  theme={"system"}
    curl -X POST http://localhost:11434/api/embed \
      -H "Content-Type: application/json" \
      -d '{
        "model": "embeddinggemma",
        "input": [
          "First sentence",
          "Second sentence",
          "Third sentence"
        ]
      }'
    ```
  </Tab>

  <Tab title="Python">
    ```python  theme={"system"}
    import ollama

    batch = ollama.embed(
      model='embeddinggemma',
      input=[
        'The quick brown fox jumps over the lazy dog.',
        'The five boxing wizards jump quickly.',
        'Jackdaws love my big sphinx of quartz.',
      ]
    )
    print(len(batch['embeddings']))  # number of vectors
    ```
  </Tab>

  <Tab title="JavaScript">
    ```javascript  theme={"system"}
    import ollama from 'ollama'

    const batch = await ollama.embed({
      model: 'embeddinggemma',
      input: [
        'The quick brown fox jumps over the lazy dog.',
        'The five boxing wizards jump quickly.',
        'Jackdaws love my big sphinx of quartz.',
      ],
    })
    console.log(batch.embeddings.length) // number of vectors
    ```
  </Tab>
</Tabs>

## Tips

* Use cosine similarity for most semantic search use cases.
* Use the same embedding model for both indexing and querying.


> ## Documentation Index
> Fetch the complete documentation index at: https://docs.ollama.com/llms.txt
> Use this file to discover all available pages before exploring further.

# Structured Outputs

Structured outputs let you enforce a JSON schema on model responses so you can reliably extract structured data, describe images, or keep every reply consistent.

## Generating structured JSON

<Tabs>
  <Tab title="cURL">
    ```shell  theme={"system"}
    curl -X POST http://localhost:11434/api/chat -H "Content-Type: application/json" -d '{
      "model": "gpt-oss",
      "messages": [{"role": "user", "content": "Tell me about Canada in one line"}],
      "stream": false,
      "format": "json"
    }'
    ```
  </Tab>

  <Tab title="Python">
    ```python  theme={"system"}
    from ollama import chat

    response = chat(
      model='gpt-oss',
      messages=[{'role': 'user', 'content': 'Tell me about Canada.'}],
      format='json'
    )
    print(response.message.content)
    ```
  </Tab>

  <Tab title="JavaScript">
    ```javascript  theme={"system"}
    import ollama from 'ollama'

    const response = await ollama.chat({
      model: 'gpt-oss',
      messages: [{ role: 'user', content: 'Tell me about Canada.' }],
      format: 'json'
    })
    console.log(response.message.content)
    ```
  </Tab>
</Tabs>

## Generating structured JSON with a schema

Provide a JSON schema to the `format` field.

<Note>
  It is ideal to also pass the JSON schema as a string in the prompt to ground the model's response.
</Note>

<Tabs>
  <Tab title="cURL">
    ```shell  theme={"system"}
    curl -X POST http://localhost:11434/api/chat -H "Content-Type: application/json" -d '{
      "model": "gpt-oss",
      "messages": [{"role": "user", "content": "Tell me about Canada."}],
      "stream": false,
      "format": {
        "type": "object",
        "properties": {
          "name": {"type": "string"},
          "capital": {"type": "string"},
          "languages": {
            "type": "array",
            "items": {"type": "string"}
          }
        },
        "required": ["name", "capital", "languages"]
      }
    }'
    ```
  </Tab>

  <Tab title="Python">
    Use Pydantic models and pass `model_json_schema()` to `format`, then validate the response:

    ```python  theme={"system"}
    from ollama import chat
    from pydantic import BaseModel

    class Country(BaseModel):
      name: str
      capital: str
      languages: list[str]

    response = chat(
      model='gpt-oss',
      messages=[{'role': 'user', 'content': 'Tell me about Canada.'}],
      format=Country.model_json_schema(),
    )

    country = Country.model_validate_json(response.message.content)
    print(country)
    ```
  </Tab>

  <Tab title="JavaScript">
    Serialize a Zod schema with `zodToJsonSchema()` and parse the structured response:

    ```javascript  theme={"system"}
    import ollama from 'ollama'
    import { z } from 'zod'
    import { zodToJsonSchema } from 'zod-to-json-schema'

    const Country = z.object({
      name: z.string(),
      capital: z.string(),
      languages: z.array(z.string()),
    })

    const response = await ollama.chat({
      model: 'gpt-oss',
      messages: [{ role: 'user', content: 'Tell me about Canada.' }],
      format: zodToJsonSchema(Country),
    })

    const country = Country.parse(JSON.parse(response.message.content))
    console.log(country)
    ```
  </Tab>
</Tabs>

## Example: Extract structured data

Define the objects you want returned and let the model populate the fields:

```python  theme={"system"}
from ollama import chat
from pydantic import BaseModel

class Pet(BaseModel):
  name: str
  animal: str
  age: int
  color: str | None
  favorite_toy: str | None

class PetList(BaseModel):
  pets: list[Pet]

response = chat(
  model='gpt-oss',
  messages=[{'role': 'user', 'content': 'I have two cats named Luna and Loki...'}],
  format=PetList.model_json_schema(),
)

pets = PetList.model_validate_json(response.message.content)
print(pets)
```

## Example: Vision with structured outputs

Vision models accept the same `format` parameter, enabling deterministic descriptions of images:

```python  theme={"system"}
from ollama import chat
from pydantic import BaseModel
from typing import Literal, Optional

class Object(BaseModel):
  name: str
  confidence: float
  attributes: str

class ImageDescription(BaseModel):
  summary: str
  objects: list[Object]
  scene: str
  colors: list[str]
  time_of_day: Literal['Morning', 'Afternoon', 'Evening', 'Night']
  setting: Literal['Indoor', 'Outdoor', 'Unknown']
  text_content: Optional[str] = None

response = chat(
  model='gemma3',
  messages=[{
    'role': 'user',
    'content': 'Describe this photo and list the objects you detect.',
    'images': ['path/to/image.jpg'],
  }],
  format=ImageDescription.model_json_schema(),
  options={'temperature': 0},
)

image_description = ImageDescription.model_validate_json(response.message.content)
print(image_description)
```

## Tips for reliable structured outputs

* Define schemas with Pydantic (Python) or Zod (JavaScript) so they can be reused for validation.
* Lower the temperature (e.g., set it to `0`) for more deterministic completions.
* Structured outputs work through the OpenAI-compatible API via `response_format`


> ## Documentation Index
> Fetch the complete documentation index at: https://docs.ollama.com/llms.txt
> Use this file to discover all available pages before exploring further.

# Thinking

Thinking-capable models emit a `thinking` field that separates their reasoning trace from the final answer.

Use this capability to audit model steps, animate the model *thinking* in a UI, or hide the trace entirely when you only need the final response.

## Supported models

* [Qwen 3](https://ollama.com/library/qwen3)
* [GPT-OSS](https://ollama.com/library/gpt-oss) *(use `think` levels: `low`, `medium`, `high` — the trace cannot be fully disabled)*
* [DeepSeek-v3.1](https://ollama.com/library/deepseek-v3.1)
* [DeepSeek R1](https://ollama.com/library/deepseek-r1)
* Browse the latest additions under [thinking models](https://ollama.com/search?c=thinking)

## Enable thinking in API calls

Set the `think` field on chat or generate requests. Most models accept booleans (`true`/`false`).

GPT-OSS instead expects one of `low`, `medium`, or `high` to tune the trace length.

The `message.thinking` (chat endpoint) or `thinking` (generate endpoint) field contains the reasoning trace while `message.content` / `response` holds the final answer.

<Tabs>
  <Tab title="cURL">
    ```shell  theme={"system"}
    curl http://localhost:11434/api/chat -d '{
      "model": "qwen3",
      "messages": [{
        "role": "user",
        "content": "How many letter r are in strawberry?"
      }],
      "think": true,
      "stream": false
    }'
    ```
  </Tab>

  <Tab title="Python">
    ```python  theme={"system"}
    from ollama import chat

    response = chat(
      model='qwen3',
      messages=[{'role': 'user', 'content': 'How many letter r are in strawberry?'}],
      think=True,
      stream=False,
    )

    print('Thinking:\n', response.message.thinking)
    print('Answer:\n', response.message.content)
    ```
  </Tab>

  <Tab title="JavaScript">
    ```javascript  theme={"system"}
    import ollama from 'ollama'

    const response = await ollama.chat({
      model: 'deepseek-r1',
      messages: [{ role: 'user', content: 'How many letter r are in strawberry?' }],
      think: true,
      stream: false,
    })

    console.log('Thinking:\n', response.message.thinking)
    console.log('Answer:\n', response.message.content)
    ```
  </Tab>
</Tabs>

<Note>
  GPT-OSS requires `think` to be set to `"low"`, `"medium"`, or `"high"`. Passing `true`/`false` is ignored for that model.
</Note>

## Stream the reasoning trace

Thinking streams interleave reasoning tokens before answer tokens. Detect the first `thinking` chunk to render a "thinking" section, then switch to the final reply once `message.content` arrives.

<Tabs>
  <Tab title="Python">
    ```python  theme={"system"}
    from ollama import chat

    stream = chat(
      model='qwen3',
      messages=[{'role': 'user', 'content': 'What is 17 × 23?'}],
      think=True,
      stream=True,
    )

    in_thinking = False

    for chunk in stream:
      if chunk.message.thinking and not in_thinking:
        in_thinking = True
        print('Thinking:\n', end='')

      if chunk.message.thinking:
        print(chunk.message.thinking, end='')
      elif chunk.message.content:
        if in_thinking:
          print('\n\nAnswer:\n', end='')
          in_thinking = False
        print(chunk.message.content, end='')

    ```
  </Tab>

  <Tab title="JavaScript">
    ```javascript  theme={"system"}
    import ollama from 'ollama'

    async function main() {
      const stream = await ollama.chat({
        model: 'qwen3',
        messages: [{ role: 'user', content: 'What is 17 × 23?' }],
        think: true,
        stream: true,
      })

      let inThinking = false

      for await (const chunk of stream) {
        if (chunk.message.thinking && !inThinking) {
          inThinking = true
          process.stdout.write('Thinking:\n')
        }

        if (chunk.message.thinking) {
          process.stdout.write(chunk.message.thinking)
        } else if (chunk.message.content) {
          if (inThinking) {
            process.stdout.write('\n\nAnswer:\n')
            inThinking = false
          }
          process.stdout.write(chunk.message.content)
        }
      }
    }

    main()
    ```
  </Tab>
</Tabs>

## CLI quick reference

* Enable thinking for a single run: `ollama run deepseek-r1 --think "Where should I visit in Lisbon?"`
* Disable thinking: `ollama run deepseek-r1 --think=false "Summarize this article"`
* Hide the trace while still using a thinking model: `ollama run deepseek-r1 --hidethinking "Is 9.9 bigger or 9.11?"`
* Inside interactive sessions, toggle with `/set think` or `/set nothink`.
* GPT-OSS only accepts levels: `ollama run gpt-oss --think=low "Draft a headline"` (replace `low` with `medium` or `high` as needed).

<Note>Thinking is enabled by default in the CLI and API for supported models.</Note>

> ## Documentation Index
> Fetch the complete documentation index at: https://docs.ollama.com/llms.txt
> Use this file to discover all available pages before exploring further.

# Streaming

Streaming allows you to render text as it is produced by the model.

Streaming is enabled by default through the REST API, but disabled by default in the SDKs.

To enable streaming in the SDKs, set the `stream` parameter to `True`.

## Key streaming concepts

1. Chatting: Stream partial assistant messages. Each chunk includes the `content` so you can render messages as they arrive.
2. Thinking: Thinking-capable models emit a `thinking` field alongside regular content in each chunk. Detect this field in streaming chunks to show or hide reasoning traces before the final answer arrives.
3. Tool calling: Watch for streamed `tool_calls` in each chunk, execute the requested tool, and append tool outputs back into the conversation.

## Handling streamed chunks

<Note> It is necessary to accumulate the partial fields in order to maintain the history of the conversation. This is particularly important for tool calling where the thinking, tool call from the model, and the executed tool result must be passed back to the model in the next request. </Note>

<Tabs>
  <Tab title="Python">
    ```python  theme={"system"}
    from ollama import chat

    stream = chat(
      model='qwen3',
      messages=[{'role': 'user', 'content': 'What is 17 × 23?'}],
      stream=True,
    )

    in_thinking = False
    content = ''
    thinking = ''
    for chunk in stream:
      if chunk.message.thinking:
        if not in_thinking:
          in_thinking = True
          print('Thinking:\n', end='', flush=True)
        print(chunk.message.thinking, end='', flush=True)
        # accumulate the partial thinking 
        thinking += chunk.message.thinking
      elif chunk.message.content:
        if in_thinking:
          in_thinking = False
          print('\n\nAnswer:\n', end='', flush=True)
        print(chunk.message.content, end='', flush=True)
        # accumulate the partial content
        content += chunk.message.content

      # append the accumulated fields to the messages for the next request
      new_messages = [{ role: 'assistant', thinking: thinking, content: content }]
    ```
  </Tab>

  <Tab title="JavaScript">
    ```javascript  theme={"system"}
    import ollama from 'ollama'

    async function main() {
      const stream = await ollama.chat({
        model: 'qwen3',
        messages: [{ role: 'user', content: 'What is 17 × 23?' }],
        stream: true,
      })

      let inThinking = false
      let content = ''
      let thinking = ''

      for await (const chunk of stream) {
        if (chunk.message.thinking) {
          if (!inThinking) {
            inThinking = true
            process.stdout.write('Thinking:\n')
          }
          process.stdout.write(chunk.message.thinking)
          // accumulate the partial thinking
          thinking += chunk.message.thinking
        } else if (chunk.message.content) {
          if (inThinking) {
            inThinking = false
            process.stdout.write('\n\nAnswer:\n')
          }
          process.stdout.write(chunk.message.content)
          // accumulate the partial content
          content += chunk.message.content
        }
      }

      // append the accumulated fields to the messages for the next request
      new_messages = [{ role: 'assistant', thinking: thinking, content: content }]
    }

    main().catch(console.error)
    ```
  </Tab>
</Tabs>

