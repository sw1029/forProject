import 'dart:convert';
import 'package:http/http.dart' as http;
import 'dart:io';
import 'dart:typed_data';
import 'package:image/image.dart' as img;
import 'package:path/path.dart' as path;


/*
* ApiService class를 통해 localhost 5000포트에서 데이터를 받아오게 구현했습니다
*
* db에 알맞은 형식의 이름을 가지는 데이터만 넣어놓고, 그걸 통해 원하는 이미지를 불러오는 쪽으로 구현하면 될 듯 합니다
* flutter 코드에는 이미지 삽입 부분은 그대로 두고,
*   이 dart 파일에서 img 저장 기능 호출만 해주는 쪽으로 구현하면 될것 같습니다
*
* <<구현한 기능 목록>>
* getRiss: RISS 상위 5개 논문의 제목과 url을 list로 반환합니다.
* getSentence: sentence table에 저장한 문장들 호출. 사전 가공한 문장을 불러올 때 쓰입니다
* analyzeEmotion: 입력한 텍스트의 분석값을 반환합니다.
*                 이진분류값과 softmax한 확률정보를 반환하는 스타일로 구현해 놨습니다.
*                 원하는 형태로 추가 가중치만 조절하면 구현은 끝납니다.
* getMonthData: 한 달 단위의 분석값을 table에서 긁어옵니다. 한달통계를 반환하게 추후 구현 가능합니다.
* getSweetPotato: 지정한 월의 고구마 이미지를 저장합니다.
* getAllSP: 월별 고구마 table의 모든 고구마 이미지를 저장합니다.
* saveImg: blob형태로 encoding된 img를 decode해주기 위한 함수입니다.
* */
class ApiService {
  static final ApiService _instance = ApiService._internal();
  factory ApiService() => _instance;
  ApiService._internal();

  // 서버 기본 URL
  static const String _baseUrl = 'http://localhost:5000';

  final Map<String, String> _headers = {
    'Content-Type': 'application/json', // 값 형식 지정
  };

  // riss 엔드포인트
  Future<List<dynamic>> getRiss(String keyword) async {
    final url = Uri.parse('$_baseUrl/riss?keyword=$keyword');
    final response = await http.get(url, headers: _headers);

    if (response.statusCode == 200) {
      final data = jsonDecode(response.body);
      return data['data']; // 제목과 URL 리스트 반환
    } else {
      throw Exception('riss 접속 실패: ${response.reasonPhrase}');
    }
  }

  // sentence 엔드포인트
  Future<String> getSentence(int id) async {
    final url = Uri.parse('$_baseUrl/sentence?id=$id');
    final response = await http.get(url, headers: _headers);

    if (response.statusCode == 200) {
      final data = jsonDecode(response.body);
      return data['data']; // 문장 반환
    } else {
      throw Exception('문장 반환 실패: ${response.reasonPhrase}');
    }
  }

  // emotion 엔드포인트
  Future<Map<String, dynamic>> analyzeEmotion(String text) async {
    final url = Uri.parse('$_baseUrl/emotion');
    final response = await http.post(
      url,
      headers: _headers,
      body: jsonEncode({'text': text}),
    );

    if (response.statusCode == 200) {
      final data = jsonDecode(response.body);
      return data; // 'case'와 'probabilities' 포함
    } else {
      throw Exception('분석값 호출 실패: ${response.reasonPhrase}');
    }
  }

  // /monthdata 엔드포인트
  Future<List<dynamic>> getMonthData(String uid, String year, String month) async {
    final url = Uri.parse('$_baseUrl/monthdata?uid=$uid&year=$year&month=$month');
    final response = await http.get(url, headers: _headers);

    if (response.statusCode == 200) {
      final data = jsonDecode(response.body);
      return data['data']; // 일별 분석값 리스트 반환
    } else {
      throw Exception('월단위 분석값 찾기 실패: ${response.reasonPhrase}');
    }
  }

  // /sweetpotato 엔드포인트
  void getSweetPotato(int caseValue) async {
    final url = Uri.parse('$_baseUrl/sweetpotato?case=$caseValue');
    final response = await http.get(url, headers: _headers);

    if (response.statusCode == 200) {
      final data = jsonDecode(response.body);
      saveImg(data['image'],data['title'] );//전달받은 이미지 저장
      return;
    } else {
      throw Exception('고구마 품절: ${response.reasonPhrase}');
    }
  }

  // /spAll 엔드포인트
  Future<Map<int, String>> getAllSP() async {
    final url = Uri.parse('$_baseUrl/spAll');
    final response = await http.get(url, headers: _headers);

    if (response.statusCode == 200) {
      final data = jsonDecode(response.body);
      List<dynamic> dataList = data['data'];
      Map<int, String> imgs = {};

      for (var item in dataList) {
        int id = item['id'];
        String title = item['title'];
        String image = item['image'];
        await saveImg(image, title);
        imgs[id] = path.join('images', title + '.png');
      }

      return imgs;
    } else {
      throw Exception('/spAll 접속 실패: ${response.reasonPhrase}');
    }
  }

  // /이미지 저장용
  Future<void> saveImg(String blob, String name) async {
    Uint8List blobData = base64Decode(blob);
    img.Image? image = img.decodeImage(blobData);
    String directoryPath = path.join(Directory.current.path, 'images');
    Directory(directoryPath).createSync(recursive: true);
    String filePath = path.join(directoryPath, name + '.png');

    if (image != null) {
      // PNG 형식으로 변환된 데이터 저장
      File(filePath)
        ..writeAsBytesSync(img.encodePng(image));
    } else {
      throw Exception('이미지 디코딩 실패');
    }
  }

}

