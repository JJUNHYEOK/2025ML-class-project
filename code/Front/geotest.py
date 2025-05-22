#/* Python 코드 사용예제 */     
import requests	
		
import requests

def get_road_address_from_coords(lat, lon):
    url = "https://nominatim.openstreetmap.org/reverse"
    params = {
        "lat": lat,
        "lon": lon,
        "format": "json",
        "zoom": 18,  # 상세 주소 레벨 (건물 단위)
        "accept-language": "ko"  # 한글 주소 요청
    }
    headers = {
        "User-Agent": "Your-App-Name/1.0"  # 필수: 사용자 식별 정보
    }
    
    try:
        response = requests.get(url, params=params, headers=headers, timeout=5)
        if response.status_code == 200:
            data = response.json()
            address = data.get('address', {})
            
            # 도로명 주소 조합 (예시)
            road = address.get('road', '')
            suburb = address.get('suburb', '')
            city = address.get('city', '')
            country = address.get('country', '')
            
            if road:
                return f"{country} {city} {suburb} {road}"
            else:
                return "주소 없음"
        else:
            return f"API 오류: {response.status_code}"
    except Exception as e:
        return f"오류: {str(e)}"

# 사용 예시 128.1128,35.1534
lat = 35.1534  # 서울시청 위도
lon = 128.1128  # 서울시청 경도
print(get_road_address_from_coords(lat, lon))
# 출력: "대한민국 서울특별시 중구 태평로1가 세종대로 110"
