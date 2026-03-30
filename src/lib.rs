mod classic;
mod common;
mod itemsets;
mod pairs;
mod pipeline;

use pyo3::prelude::*;

#[pymodule]
fn _fastapriori_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(pairs::rust_compute_pairs, m)?)?;
    m.add_function(wrap_pyfunction!(itemsets::rust_compute_itemsets, m)?)?;
    m.add_function(wrap_pyfunction!(pipeline::rust_compute_pipeline, m)?)?;
    m.add_function(wrap_pyfunction!(classic::rust_classic_compute_pairs, m)?)?;
    m.add_function(wrap_pyfunction!(classic::rust_classic_compute_pipeline, m)?)?;
    Ok(())
}
