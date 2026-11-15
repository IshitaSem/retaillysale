from typing import Dict, Any, Optional
from pydantic import BaseModel, Field

class PredictionRequest(BaseModel):
    unit_price: float = Field(..., gt=0, description="Unit price in INR (e.g. 40.98)")
    qty_ordered: int = Field(..., ge=1, le=500, description="Quantity ordered (e.g. 5)")
    discount_offered: float = Field(0.05, ge=0.0, le=1.0, description="Discount fraction (e.g. 0.05 for 5%)")
    freight_expenses: float = Field(6.05, ge=0.0, description="Freight cost in INR (e.g. 6.05)")
    shipping_days: int = Field(2, ge=0, le=180, description="Estimated transit days (e.g. 2)")
    order_month: int = Field(6, ge=1, le=12, description="Order month (1-12)")
    order_day_of_week: str = Field("Wednesday", description="Day of week (Monday..Sunday)")
    order_priority: str = Field("High", description="Order priority: Critical, High, Medium, Low, Not Specified")
    freight_mode: str = Field("Regular Air", description="Logistics mode: Regular Air, Express Air, Delivery Truck")
    segment: str = Field("Restaurant Chain", description="Customer segment: Restaurant Chain, Hotels / Hospitals, Stand Alone Restaurants, Personel Usage")
    product_type: str = Field("Processed Meat", description="Product category: Processed Meat, Canned Foods, Preserved Food")
    product_container: str = Field("Small Box", description="Container: Small Box, Wrap Bag, Large Box, Jumbo Drum, Medium Box, Small Pack, Jumbo Box")
    region: str = Field("West", description="Geographic region: West, East, North, South")
    model_type: str = Field("Random Forest", description="Model choice: Random Forest or Linear Regression")
    use_selected_features: bool = Field(True, description="Whether to use the top-k selected feature pipeline")

class PredictionResponse(BaseModel):
    predicted_sales: float
    predicted_sales_formatted: str
    model_used: str
    feature_subset: str
    inputs: Dict[str, Any]
    explanation: str
