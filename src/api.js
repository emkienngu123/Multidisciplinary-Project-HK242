// src/api.js
const BASE = 'https://smart-home-be-1.onrender.com/api'

// lấy token từ localStorage
function getToken() {
  return localStorage.getItem('jwtToken')
}

// helper để gửi request
async function request(path, { method = 'GET', body, qs, includeToken = true } = {}) {
  let url = `${BASE}${path}`
  if (qs) {
    const params = new URLSearchParams(qs).toString()
    url += `?${params}`
  }

  const headers = { 'Content-Type': 'application/json' }
  if (includeToken) {
    const token = getToken()
    if (token) headers['Authorization'] = `Bearer ${token}`
  }

  const res = await fetch(url, {
    method,
    headers,
    credentials: 'include',
    body: body ? JSON.stringify(body) : undefined
  })
  const json = await res.json()
  if (!res.ok) throw new Error(json.message || res.statusText)
  return json.data
}

export function signUp(username, password) {
  return request('/signup', { method: 'POST', includeToken: false, body: { username, password } })
}

export async function signIn(username, password) {
  const data = await request('/signin', { method: 'POST', includeToken: false, body: { username, password } })
  // lưu token
  localStorage.setItem('jwtToken', data.accessToken)
  return data
}

// GET endpoints
export function getSensors() {
  return request('/sensor')
}
export function getTemperature(numRecord = 3) {
  return request('/temperature', { qs: { numRecord } })
}
export function getHumidity(numRecord = 3) {
  return request('/humidity', { qs: { numRecord } })
}
export function getMovement(numRecord = 3) {
  return request('/movement', { qs: { numRecord } })
}
export function getLight(numRecord = 3) {
  return request('/light', { qs: { numRecord } })
}
export function getFanSpeed() {
  return request('/fanspeed')
}
export function getLightIntensity() {
  return request('/lightintensity')
}
export function getAlerts() {
  return request('/alert')
}

// PATCH endpoints
export function setFanSpeed(speed) {
  return request('/fanspeed', { method: 'PATCH', body: { speed } })
}
export function setLightIntensity(value) {
  return request('/lightintensity', { method: 'PATCH', body: { value } })
}
export function setAutoAdjust(enable) {
  return request('/autoadjust', { method: 'PATCH', qs: { enable } })
}

// Voice command
export async function voiceCommand(file) {
  const token = getToken()
  const form = new FormData()
  form.append('audio', file)
  const res = await fetch(`${BASE}/voicecommand`, {
    method: 'POST',
    headers: { 'Authorization': `Bearer ${token}` },
    body: form
  })
  const json = await res.json()
  if (!res.ok) throw new Error(json.message)
  return json.data
}


export async function faceSignIn(formData) {
  const res = await fetch(`${BASE}/facesignin`, {
    method: 'POST',
    body: formData
  });
  const json = await res.json();
  if (!res.ok) throw new Error(json.message || res.statusText);
  return json.data;
}
