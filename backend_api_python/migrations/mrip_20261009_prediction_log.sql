-- MRIP automatic prediction logging (work 023). Idempotent; applied as a bootstrap component by
-- app/utils/db.py. Adds the two automatically logged prediction types. Value list must match
-- app/mrip/outcomes/types.py PredictionType (enforced by tests).
--
-- Discriminators (which discover kind, which related edge) are carried in model_version, so the
-- existing unique key (type, subject, made_at, horizon, model_version) dedupes them without a
-- schema change.

ALTER TABLE mrip_predictions DROP CONSTRAINT IF EXISTS mrip_predictions_prediction_type_check;
ALTER TABLE mrip_predictions ADD CONSTRAINT mrip_predictions_prediction_type_check
    CHECK (prediction_type IN ('forecast', 'options_event', 'discover_item', 'related_signal'));
