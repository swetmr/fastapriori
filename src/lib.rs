mod classic;
mod common;
mod eclat;
mod itemsets;
mod pairs;
mod pipeline;

use pyo3::prelude::*;

#[pymodule]
fn _fastapriori_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(pairs::rust_compute_pairs, m)?)?;
    m.add_function(wrap_pyfunction!(itemsets::rust_compute_itemsets, m)?)?;
    m.add_function(wrap_pyfunction!(pipeline::rust_compute_pipeline, m)?)?;
    m.add_function(wrap_pyfunction!(pipeline::rust_compute_pipeline_v1_roaring, m)?)?;
    m.add_function(wrap_pyfunction!(pipeline::rust_compute_pipeline_v2_memo, m)?)?;
    m.add_function(wrap_pyfunction!(pipeline::rust_compute_pipeline_v3_adaptive, m)?)?;
    m.add_function(wrap_pyfunction!(pipeline::rust_compute_pipeline_v5_prefilter, m)?)?;
    m.add_function(wrap_pyfunction!(pipeline::rust_compute_pipeline_v6_gating, m)?)?;
    m.add_function(wrap_pyfunction!(eclat::rust_eclat_pipeline, m)?)?;
    m.add_function(wrap_pyfunction!(classic::rust_classic_compute_pairs, m)?)?;
    m.add_function(wrap_pyfunction!(classic::rust_classic_compute_pipeline, m)?)?;
    Ok(())
}
