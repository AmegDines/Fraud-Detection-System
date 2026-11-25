"""
Financial Fraud Detection & Fraud Ring Analysis Platform
Modules:
- producer: Kafka transaction generator
- batch_pipeline: Spark SQL aggregations, windowing, and velocity checks
- graph_analysis: Low-level RDDs & Graph Analytics (Connected Components / Fraud Rings)
- ml_trainer: Spark MLlib classification & performance evaluation
- streaming_pipeline: Spark Structured Streaming live scoring sink
"""
__version__ = "1.0.0"
