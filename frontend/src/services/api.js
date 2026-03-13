import axios from "axios"

const API_BASE = axios.create({
  baseURL: "http://localhost:8000",
})

export const getBrandAnalysis = (brand) => {
    return API_BASE.get(`/colours/brand/${brand}`)
}