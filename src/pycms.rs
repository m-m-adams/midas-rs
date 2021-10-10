use super::cms::*;
use pyo3::prelude::*;

#[pyclass]
#[derive(Clone)]
pub struct PyCMS(CMS<i64>);

#[pymethods]
impl PyCMS {
    #[new]
    fn new(tol: f64, err: f64, capacity: usize) -> Self {
        PyCMS(CMS::new_with_probs(tol, err, capacity))
    }
    fn insert(&mut self, edge: PyObject, py: Python) -> PyResult<u64> {
        let hash: i64 = edge.call_method0(py, "__hash__")?.extract(py)?;
        Ok(self.0.insert(hash))
    }
    fn retrieve(&mut self, edge: PyObject, py: Python) -> PyResult<u64> {
        let hash: i64 = edge.call_method0(py, "__hash__")?.extract(py)?;
        Ok(self.0.retrieve(hash))
    }
    fn clear(&mut self) {
        self.0.clear()
    }
    fn scale(&mut self, factor: f64) {
        self.0.scale(factor)
    }

    fn combine(&mut self, other: PyCMS) -> PyResult<()> {
        Ok(self.0.combine(&other.0)?)
    }
}

#[pymodule]
fn count_min_sketch(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyCMS>()?;

    #[pyfn(m)]
    fn clone_cms(base: PyCMS) -> PyCMS {
        PyCMS(CMS::new_from_cms(&base.0))
    }

    Ok(())
}
