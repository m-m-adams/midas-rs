use super::cms::*;
use pyo3::prelude::*;

#[pyclass]
#[derive(Clone)]
pub struct CMS(CountMinSketch<isize>);

struct Edge(isize);

impl<'source> FromPyObject<'source> for Edge {
    fn extract(ob: &'source PyAny) -> PyResult<Self> {
        let tryint: Result<isize, PyErr> = ob.extract();
        match tryint {
            Ok(i) => return Ok(Edge(i)),
            Err(_) => {
                let hash: isize = ob.hash()?;
                return Ok(Edge(hash));
            }
        }
    }
}

#[pymethods]
impl CMS {
    #[new]
    fn new(depth: u64, width: u64) -> Self {
        CMS(CountMinSketch::new(depth, width))
    }
    fn insert(&mut self, e: Edge) -> PyResult<u64> {
        //let hash: i64 = edge.call_method0(py, "__hash__")?.extract(py)?;

        //let hash: i64 = edge.extract();
        Ok(self.0.insert(e.0))
    }
    fn retrieve(&mut self, e: Edge) -> PyResult<u64> {
        Ok(self.0.retrieve(e.0))
    }
    fn clear(&mut self) {
        self.0.clear()
    }
    fn scale(&mut self, factor: f64) {
        self.0.scale(factor)
    }

    fn combine(&mut self, other: CMS) -> PyResult<()> {
        Ok(self.0.combine(&other.0)?)
    }
}

#[pymodule]
fn count_min_sketch(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<CMS>()?;

    #[pyfn(m)]
    fn clone_cms(base: CMS) -> CMS {
        CMS(CountMinSketch::new_from_cms(&base.0))
    }

    Ok(())
}
