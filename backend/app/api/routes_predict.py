from fastapi import APIRouter, HTTPException
from ..ml.predictor import SalesPredictor
from ..schemas.predict import PredictionRequest, PredictionResponse

router = APIRouter(prefix="/api", tags=["Prediction"])

@router.post("/predict", response_model=PredictionResponse, summary="Predict retail sales")
def predict_sales(payload: PredictionRequest):
    """Predicts retail sales based on transaction inputs using the trained machine learning pipeline."""
    try:
        result = SalesPredictor.predict(
            unit_price=payload.unit_price,
            qty_ordered=payload.qty_ordered,
            discount_offered=payload.discount_offered,
            freight_expenses=payload.freight_expenses,
            shipping_days=payload.shipping_days,
            order_month=payload.order_month,
            order_day_of_week=payload.order_day_of_week,
            order_priority=payload.order_priority,
            freight_mode=payload.freight_mode,
            segment=payload.segment,
            product_type=payload.product_type,
            product_container=payload.product_container,
            region=payload.region,
            model_type=payload.model_type,
            use_selected_features=payload.use_selected_features
        )
        return result
    except ValueError as ve:
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction execution failed: {str(e)}")
