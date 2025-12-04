import requests
import json
import sys

BASE_URL = "http://localhost:8000"


def test_evaluation(ub_id: int):
    print(f"\n{'='*70}")
    print(f" Тестування Evaluation для UB ID: {ub_id}")
    print(f"{'='*70}\n")
    
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=2)
        if response.status_code != 200:
            print("❌ Сервер не відповідає. Запустіть main.py спочатку!")
            return False
    except:
        print("❌ Сервер не запущений. Запустіть: python main.py")
        return False
    
    print("✅ Сервер запущений\n")
    
    print("📚 Завантаження workflow state...")
    try:
        state_response = requests.get(f"{BASE_URL}/chat/{ub_id}/state")
        if state_response.status_code == 200:
            state_data = state_response.json()
            answers_count = len(state_data.get('answers', []))
            questions_count = len(state_data.get('questions', []))
            status = state_data.get('status', 'unknown')
            
            print(f"   ✅ Стан знайдено")
            print(f"   📊 Статус: {status}")
            print(f"   💬 Питань: {questions_count}")
            print(f"   ✍️  Відповідей: {answers_count}")
            
            if answers_count == 0:
                print("\n⚠️  Немає відповідей для оцінювання.")
                print(f"   Спочатку відправте повідомлення:")
                print(f"   python test_agents.py {ub_id} 'ваша відповідь'")
                return False
            
            print("\n📝 Останні відповіді:")
            for i, ans in enumerate(state_data.get('answers', [])[-3:], 1):
                answer_text = ans.get('answer', 'немає відповіді')
                evaluation = ans.get('evaluation', {})
                complete = evaluation.get('complete', False)
                
                print(f"\n  {i}. Відповідь: {answer_text[:80]}...")
                print(f"     Повна: {complete}")
                if not complete and evaluation.get('missing_concepts'):
                    print(f"     Не вистачає: {evaluation.get('missing_concepts')}")
        
        elif state_response.status_code == 404:
            print("❌ Workflow state не знайдено.")
            print(f"   Спочатку почніть чат:")
            print(f"   python test_agents.py {ub_id} 'привіт'")
            return False
        else:
            print(f"❌ Помилка завантаження state: {state_response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Помилка: {e}")
        return False
    
    print(f"\n{'='*70}")
    print("🔬 Запуск оцінювання...")
    print(f"{'='*70}\n")
    
    try:
        import time
        start_time = time.time()
        
        response = requests.post(
            f"{BASE_URL}/chat/{ub_id}/evaluate",
            timeout=120
        )
        
        elapsed = time.time() - start_time
        
        print(f"⏱️  Час виконання: {elapsed:.2f}s")
        print(f"📊 Статус відповіді: {response.status_code}\n")
        
        if response.status_code == 200:
            data = response.json()
            
            print("✅ Оцінювання успішно завершено!\n")
            print(f"{'='*70}")
            print(" РЕЗУЛЬТАТ ОЦІНЮВАННЯ")
            print(f"{'='*70}\n")
            
            evaluation_text = data.get('evaluation', '')
            timestamp = data.get('timestamp', '')
            conversation_length = data.get('conversation_length', 0)
            criteria_count = data.get('criteria_count', 0)
            cached = data.get('cached', False)
            grade_saved = data.get('grade_saved', False)
            
            print(f"🕐 Час: {timestamp}")
            print(f"💬 Відповідей проаналізовано: {conversation_length}")
            print(f"📊 Критеріїв оцінювання: {criteria_count}")
            print(f"💾 Збережено в Xano (update_ub): {'✅ Так' if grade_saved else '❌ Ні'}")
            print(f"🔄 З кешу: {'Так' if cached else 'Ні'}\n")
            print("📋 Оцінка:\n")
            print(evaluation_text)
            print(f"\n{'='*70}\n")
            
            if grade_saved:
                print("✅ Оцінка успішно збережена в таблицю UB через update_ub endpoint")
            else:
                print("⚠️  Оцінка НЕ збережена. Перевірте:")
                print("   1. Чи правильний XANO_BASE_URL в .env")
                print("   2. Чи існує endpoint /update_ub на Xano")
                print("   3. Логи сервера для деталей помилки")
            
            return True
            
        elif response.status_code == 400:
            error_data = response.json()
            print(f"⚠️  {error_data.get('detail', 'Невідома помилка')}")
            print("\n💡 Можливо, для цього блоку не налаштований evaluation.")
            return False
        
        elif response.status_code == 404:
            error_data = response.json()
            print(f"⚠️  {error_data.get('detail', 'Невідома помилка')}")
            print(f"\n💡 Спочатку відправте відповіді:")
            print(f"   python test_agents.py {ub_id} 'ваша відповідь'")
            return False
            
        else:
            print(f"❌ Помилка {response.status_code}")
            print(f"Відповідь: {response.text[:500]}")
            return False
            
    except requests.Timeout:
        print("❌ Таймаут (>120s). Оцінювання займає занадто багато часу.")
        return False
    except Exception as e:
        print(f"❌ Помилка: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_xano_grade(ub_id: int):
    print(f"\n{'='*70}")
    print(" ПЕРЕВІРКА ЗБЕРЕЖЕННЯ В XANO")
    print(f"{'='*70}\n")
    
    print(f"💡 Перевірте в Xano Dashboard:")
    print(f"   1. Відкрийте таблицю 'UB'")
    print(f"   2. Знайдіть запис з id = {ub_id}")
    print(f"   3. Перевірте поля:")
    print(f"      - grade (текст оцінки)")
    print(f"      - status (має бути 'finished')")
    print(f"      - grading_output (JSON з результатами)")
    print(f"      - summary_timestamp (час оцінювання)")
    print(f"\n   URL: https://your-workspace.xano.io/admin/table/ub/{ub_id}")


def main():
    print("\n🎓 EdTech AI Platform - Тестування Evaluation з Xano Integration\n")
    
    if len(sys.argv) < 2:
        print("❌ Використання: python test_evaluation.py <UB_ID>")
        print("\nПриклад:")
        print("  python test_evaluation.py 12610")
        print("\nПослідовність тестування:")
        print("  1. python test_agents.py 12610 'Ембединг'")
        print("  2. python test_agents.py 12610 'Позиційне кодування'")
        print("  3. python test_evaluation.py 12610")
        sys.exit(1)
    
    try:
        ub_id = int(sys.argv[1])
    except ValueError:
        print("❌ UB_ID має бути числом")
        sys.exit(1)
    
    success = test_evaluation(ub_id)
    
    if success:
        verify_xano_grade(ub_id)
        print("\n✅ Тест успішно завершено!")
        print("\n💡 Наступні кроки:")
        print(f"   - Перевірте Xano таблицю UB для запису {ub_id}")
        print(f"   - Подивіться логи update_ub endpoint в Xano")
        print(f"   - Запустіть ще раз для перевірки кешування:")
        print(f"     python test_evaluation.py {ub_id}")
    else:
        print("\n❌ Тест завершився з помилками")
        sys.exit(1)


if __name__ == "__main__":
    main()